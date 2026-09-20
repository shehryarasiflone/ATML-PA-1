import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from common.seed import set_seed
from task2.models.backbone import PACSResNet18, freeze_bn_stats
from task2.data.pacs_loader import get_pacs_dataloaders
from task2.scripts.train_source_only import evaluate_domain
from task3.methods.sam import SAM
from task3.scripts.evaluate_task3 import get_fixed_sharpness_batch, compute_sharpness_proxy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_sam_instance(rho: float):
    set_seed(6304)
    src_train_loaders, _, src_val_loaders, _ = get_pacs_dataloaders()

    model = PACSResNet18(num_classes=7).to(DEVICE)
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=rho,
        lr=1e-4,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    num_batches = max(len(loader) for loader in src_train_loaders.values())
    best_val_f1 = -1.0
    patience = 5
    patience_counter = 0

    os.makedirs("task3/checkpoints/study", exist_ok=True)
    checkpoint_path = f"task3/checkpoints/study/sam_rho_{rho}_best.pth"

    # Check if this setting was already trained in Step 3
    if rho == 0.05 and os.path.exists("task3/checkpoints/sam_rho_0.05_best.pth"):
        print(f"Reusing existing checkpoint for ρ = {rho} from Step 3.")
        return "task3/checkpoints/sam_rho_0.05_best.pth"

    print(f"\n--- Training SAM with ρ = {rho} ---")

    for epoch in range(1, 31):
        model.train()
        freeze_bn_stats(model)

        iter_loaders = {d: iter(loader) for d, loader in src_train_loaders.items()}
        epoch_loss = 0.0

        for _ in range(num_batches):
            batch_imgs, batch_labels = [], []
            for d, loader in src_train_loaders.items():
                try:
                    imgs, labels, _ = next(iter_loaders[d])
                except StopIteration:
                    iter_loaders[d] = iter(loader)
                    imgs, labels, _ = next(iter_loaders[d])
                batch_imgs.append(imgs)
                batch_labels.append(labels)

            x_src = torch.cat(batch_imgs, dim=0).to(DEVICE)
            y_src = torch.cat(batch_labels, dim=0).to(DEVICE)

            # Pass 1: Ascent
            freeze_bn_stats(model)
            _, logits = model(x_src)
            loss1 = criterion(logits, y_src)
            loss1.backward()
            optimizer.first_step(zero_grad=True)

            # Pass 2: Descent
            freeze_bn_stats(model)
            _, logits_pert = model(x_src)
            loss2 = criterion(logits_pert, y_src)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            epoch_loss += loss1.item()

        # Validation across source domains
        val_f1s = []
        for d, loader in src_val_loaders.items():
            _, f1 = evaluate_domain(model, loader)
            val_f1s.append(f1)
        mean_val_f1 = np.mean(val_f1s)

        print(f"Epoch {epoch:02d} | Loss: {epoch_loss/num_batches:.4f} | Mean Source Val F1: {mean_val_f1:.2f}%")

        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    return checkpoint_path

def run_sam_study():
    set_seed(6304)
    _, _, src_val_loaders, target_eval_loader = get_pacs_dataloaders()
    sharp_x, sharp_y = get_fixed_sharpness_batch()

    rhos = [0.01, 0.05, 0.1]
    results = []

    for rho in rhos:
        cp_path = train_sam_instance(rho)

        model = PACSResNet18(num_classes=7).to(DEVICE)
        model.load_state_dict(torch.load(cp_path, map_location=DEVICE))

        # 1. Mean Source Val F1
        val_f1s = [evaluate_domain(model, src_val_loaders[d])[1] for d in ["photo", "art_painting", "cartoon"]]
        mean_src_f1 = np.mean(val_f1s)

        # 2. Sharpness Proxy (using standardized rho=0.05 perturbation)
        sharpness = compute_sharpness_proxy(model, sharp_x, sharp_y, rho=0.05)

        # 3. Sketch Target Performance
        tgt_acc, tgt_f1 = evaluate_domain(model, target_eval_loader)

        results.append({
            "rho": rho,
            "src_f1": mean_src_f1,
            "sharpness": sharpness,
            "tgt_acc": tgt_acc,
            "tgt_f1": tgt_f1
        })

    print("\n" + "="*85)
    print("TASK 3: SAM CONTROLLED DESIGN STUDY (Varying Perturbation Radius ρ)")
    print("="*85)
    print(f"{'ρ (Radius)':<12} | {'Source Val F1':<15} | {'Sharpness Proxy':<18} | {'Sketch Acc':<12} | {'Sketch F1'}")
    print("-" * 85)
    for res in results:
        print(f"{res['rho']:<12.2f} | {res['src_f1']:>13.2f}% | {res['sharpness']:>16.4f} | {res['tgt_acc']:>10.2f}% | {res['tgt_f1']:>7.2f}%")
    print("="*85)

if __name__ == "__main__":
    run_sam_study()