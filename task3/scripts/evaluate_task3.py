import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from common.pacs import PACSDataset, PACS_CLASSES, PACS_DOMAINS
from common.seed import set_seed
from task2.data.pacs_loader import EVAL_TRANSFORM, get_pacs_dataloaders
from task2.models.backbone import PACSResNet18, freeze_bn_stats
from task2.scripts.evaluate_final import extract_features_and_preds
from task2.scripts.train_source_only import evaluate_domain

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_fixed_sharpness_batch(split_path="./shared/splits/pacs_sketch_seed6304.json", n_per_domain=32):
    """
    Constructs a fixed validation batch of 32 examples from each source domain (96 total)
    using seed 6304.
    """
    with open(split_path, "r") as f:
        manifest = json.load(f)

    rng = np.random.RandomState(6304)
    all_imgs, all_labels = [], []

    for d in ["photo", "art_painting", "cartoon"]:
        val_samples = manifest["sources"][d]["val"]
        chosen_indices = rng.choice(len(val_samples), size=n_per_domain, replace=False)
        for idx in chosen_indices:
            path, label, _ = val_samples[idx]
            img = Image.open(path).convert("RGB")
            img_t = EVAL_TRANSFORM(img)
            all_imgs.append(img_t)
            all_labels.append(label)

    batch_x = torch.stack(all_imgs).to(DEVICE)
    batch_y = torch.tensor(all_labels, dtype=torch.long).to(DEVICE)
    return batch_x, batch_y

def compute_sharpness_proxy(model, batch_x, batch_y, rho=0.05):
    """
    Computes local sharpness proxy:
    Delta_sharp = L_val(theta + epsilon) - L_val(theta)
    with epsilon = rho * grad / ||grad||_2
    """
    model.eval()
    freeze_bn_stats(model)
    model.zero_grad()

    criterion = nn.CrossEntropyLoss()
    _, logits = model(batch_x)
    loss_orig = criterion(logits, batch_y)
    loss_orig.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)

    scale = rho / (grad_norm + 1e-12)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(p.grad * scale)

    with torch.no_grad():
        _, logits_pert = model(batch_x)
        loss_pert = criterion(logits_pert, batch_y)

    # Restore parameters
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.sub_(p.grad * scale)

    model.zero_grad()
    return (loss_pert - loss_orig).item()

def evaluate_all_dg():
    set_seed(6304)
    _, _, src_val_loaders, target_eval_loader = get_pacs_dataloaders()
    sharp_x, sharp_y = get_fixed_sharpness_batch()

    models_info = {
        "ERM (Baseline)": "task2/checkpoints/source_only_best.pth",
        "DAN-DG (λ=1.0)": "task3/checkpoints/dan_dg_lambda_1.0_best.pth",
        "SAM (ρ=0.05)": "task3/checkpoints/sam_rho_0.05_best.pth"
    }

    model = PACSResNet18(num_classes=7).to(DEVICE)
    results = {}
    base_sketch_acc = None

    print("\n" + "="*115)
    print("TASK 3: DOMAIN GENERALIZATION FINAL BENCHMARK")
    print("="*115)
    header = (f"{'Method':<16} | {'Photo':<9} | {'Art':<9} | {'Cartoon':<9} | "
              f"{'Mean Src':<10} | {'Worst Src':<11} | {'Sketch Acc':<11} | "
              f"{'Sketch F1':<10} | {'Δ Sketch':<9} | {'Src Sep':<8} | {'Sharpness'}")
    print(header)
    print("-" * 115)

    for name, cp_path in models_info.items():
        if not os.path.exists(cp_path):
            print(f"{name:<16} | Checkpoint missing at {cp_path}")
            continue

        model.load_state_dict(torch.load(cp_path, map_location=DEVICE))

        # 1. Source domain validation metrics
        src_val_metrics = {}
        for d in ["photo", "art_painting", "cartoon"]:
            acc, f1 = evaluate_domain(model, src_val_loaders[d])
            src_val_metrics[d] = (acc, f1)

        mean_src_acc = np.mean([m[0] for m in src_val_metrics.values()])
        mean_src_f1 = np.mean([m[1] for m in src_val_metrics.values()])

        # Worst domain identification (based on Macro-F1)
        worst_domain = min(src_val_metrics, key=lambda k: src_val_metrics[k][1])
        worst_src_acc, worst_src_f1 = src_val_metrics[worst_domain]

        # 2. Sketch Target evaluation
        sketch_acc, sketch_f1 = evaluate_domain(model, target_eval_loader)
        if name == "ERM (Baseline)":
            base_sketch_acc = sketch_acc
        delta_sketch = sketch_acc - base_sketch_acc

        # 3. Source-Domain Separability (Multinomial Logistic Regression over Photo, Art, Cartoon)
        feats_dict = {}
        for d in ["photo", "art_painting", "cartoon"]:
            f, _, _ = extract_features_and_preds(model, src_val_loaders[d], is_source_dict=False)
            feats_dict[d] = f

        min_samples = min(len(f) for f in feats_dict.values())
        rng = np.random.RandomState(6304)

        X_list, y_list = [], []
        for dom_idx, d in enumerate(["photo", "art_painting", "cartoon"]):
            idx = rng.choice(len(feats_dict[d]), min_samples, replace=False)
            X_list.append(feats_dict[d][idx])
            y_list.append(np.full(min_samples, dom_idx))

        X_src = np.vstack(X_list)
        y_src = np.concatenate(y_list)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_src, y_src, test_size=0.3, random_state=6304, stratify=y_src
        )
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=6304)
        clf.fit(X_tr, y_tr)
        sep_score = clf.score(X_te, y_te) * 100.0

        # 4. Local Sharpness Proxy
        sharp_val = compute_sharpness_proxy(model, sharp_x, sharp_y, rho=0.05)

        # Store predictions on Sketch for per-class analysis
        _, p_tgt, y_tgt = extract_features_and_preds(model, target_eval_loader, is_source_dict=False)
        results[name] = {"preds": p_tgt, "targets": y_tgt}

        print(f"{name:<16} | "
              f"{src_val_metrics['photo'][0]:>5.1f}% | "
              f"{src_val_metrics['art_painting'][0]:>5.1f}% | "
              f"{src_val_metrics['cartoon'][0]:>5.1f}% | "
              f"{mean_src_acc:>5.2f}% | "
              f"{worst_domain[:3]}({worst_src_acc:>4.1f}%) | "
              f"{sketch_acc:>6.2f}%  | "
              f"{sketch_f1:>5.2f}% | "
              f"{delta_sketch:>+6.2f}% | "
              f"{sep_score:>5.1f}%  | "
              f"{sharp_val:>8.4f}")

    print("="*115)

    # Per-Class Analysis for DG
    if "ERM (Baseline)" in results:
        print("\n--- Class-Level Analysis on Sketch (vs ERM) ---")
        y_true = results["ERM (Baseline)"]["targets"]
        p_erm = results["ERM (Baseline)"]["preds"]

        for model_name in ["DAN-DG (λ=1.0)", "SAM (ρ=0.05)"]:
            if model_name not in results:
                continue
            print(f"\nEvaluating: {model_name}")
            p_model = results[model_name]["preds"]

            class_deltas = {}
            for c_idx, c_name in enumerate(PACS_CLASSES):
                mask = (y_true == c_idx)
                acc_erm = np.mean(p_erm[mask] == c_idx) * 100
                acc_mod = np.mean(p_model[mask] == c_idx) * 100
                delta = acc_mod - acc_erm
                class_deltas[c_name] = delta
                print(f"  Class {c_name:<10}: ERM {acc_erm:>5.1f}% -> Model {acc_mod:>5.1f}% ({delta:>+5.1f}%)")

            best_cls = max(class_deltas, key=class_deltas.get)
            worst_cls = min(class_deltas, key=class_deltas.get)
            print(f"  >> Most improved: {best_cls} ({class_deltas[best_cls]:+.2f}%)")
            print(f"  >> Most degraded: {worst_cls} ({class_deltas[worst_cls]:+.2f}%)")

if __name__ == "__main__":
    evaluate_all_dg()