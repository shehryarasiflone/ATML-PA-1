import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

from common.seed import set_seed
from task2.models.backbone import PACSResNet18
from task2.data.pacs_loader import get_pacs_dataloaders
from common.pacs import PACS_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def extract_features_and_preds(model, loaders, is_source_dict=False):
    model.eval()
    all_feats, all_preds, all_labels = [], [], []
    
    if is_source_dict:
        for loader in loaders.values():
            for x, y, _ in loader:
                feat, logits = model(x.to(DEVICE))
                all_feats.append(feat.cpu())
                all_preds.append(torch.argmax(logits, dim=-1).cpu())
                all_labels.append(y)
    else:
        for x, y, _ in loaders:
            feat, logits = model(x.to(DEVICE))
            all_feats.append(feat.cpu())
            all_preds.append(torch.argmax(logits, dim=-1).cpu())
            all_labels.append(y)
            
    return torch.cat(all_feats).numpy(), torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

def evaluate_all():
    set_seed(6304)
    _, _, src_val_loaders, tgt_eval_loader = get_pacs_dataloaders()
    
    checkpoints = {
        "Source-only": "task2/checkpoints/source_only_best.pth",
        "DAN": "task2/checkpoints/dan_best.pth",
        "DANN": "task2/checkpoints/dann_best.pth",
        "CDAN": "task2/checkpoints/cdan_best.pth"
    }
    
    model = PACSResNet18(num_classes=7).to(DEVICE)
    results = {}
    
    print("="*100)
    print(f"{'Method':<15} | {'Src Val Acc':<12} | {'Src Val F1':<12} | {'Tgt Acc':<10} | {'Tgt F1':<10} | {'Δ Tgt Acc':<10} | {'Domain Sep'}")
    print("="*100)
    
    base_tgt_acc = None
    
    for name, cp_path in checkpoints.items():
        if not os.path.exists(cp_path):
            print(f"{name:<15} | Checkpoint missing at {cp_path}")
            continue
            
        model.load_state_dict(torch.load(cp_path, map_location=DEVICE))
        
        # Extract source validation and target evaluation data
        f_src, p_src, y_src = extract_features_and_preds(model, src_val_loaders, is_source_dict=True)
        f_tgt, p_tgt, y_tgt = extract_features_and_preds(model, tgt_eval_loader, is_source_dict=False)
        
        # Basic Metrics
        src_acc = np.mean(p_src == y_src) * 100
        src_f1 = f1_score(y_src, p_src, average="macro") * 100
        tgt_acc = np.mean(p_tgt == y_tgt) * 100
        tgt_f1 = f1_score(y_tgt, p_tgt, average="macro") * 100
        
        if name == "Source-only":
            base_tgt_acc = tgt_acc
        delta_tgt = (tgt_acc - base_tgt_acc) if base_tgt_acc is not None else 0.0
        
        # Domain Separability Score
        # 1. Collect equal numbers of source and target features
        min_samples = min(len(f_src), len(f_tgt))
        idx_src = np.random.choice(len(f_src), min_samples, replace=False)
        idx_tgt = np.random.choice(len(f_tgt), min_samples, replace=False)
        
        X_domain = np.vstack([f_src[idx_src], f_tgt[idx_tgt]])
        # Labels: 0 for Source, 1 for Target
        y_domain = np.concatenate([np.zeros(min_samples), np.ones(min_samples)])
        
        # 2. Train 70/30 balanced logistic regression
        X_tr, X_te, y_tr, y_te = train_test_split(X_domain, y_domain, test_size=0.3, random_state=6304, stratify=y_domain)
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
        clf.fit(X_tr, y_tr)
        sep_score = clf.score(X_te, y_te) * 100
        
        results[name] = {
            "p_tgt": p_tgt,
            "y_tgt": y_tgt
        }
        
        print(f"{name:<15} | {src_acc:>11.2f}% | {src_f1:>11.2f}% | {tgt_acc:>9.2f}% | {tgt_f1:>9.2f}% | {delta_tgt:>+9.2f}% | {sep_score:>9.2f}%")
        
    print("="*100)
    
    # Per-Class Analysis (Comparing CDAN to Source-only)
    if "Source-only" in results and "CDAN" in results:
        print("\n--- Per-Class Target Analysis (CDAN vs Source-only) ---")
        y_true = results["Source-only"]["y_tgt"]
        p_so = results["Source-only"]["p_tgt"]
        p_cdan = results["CDAN"]["p_tgt"]
        
        class_deltas = {}
        for c_idx, c_name in enumerate(PACS_CLASSES):
            mask = (y_true == c_idx)
            acc_so = np.mean(p_so[mask] == c_idx) * 100
            acc_cdan = np.mean(p_cdan[mask] == c_idx) * 100
            class_deltas[c_name] = acc_cdan - acc_so
            
        best_class = max(class_deltas, key=class_deltas.get)
        worst_class = min(class_deltas, key=class_deltas.get)
        
        print(f"Largest Improvement: {best_class} ({class_deltas[best_class]:+.2f}%)")
        print(f"Largest Degradation: {worst_class} ({class_deltas[worst_class]:+.2f}%)")
        
        # Inspect dominant confusion for the degraded class in CDAN
        worst_idx = PACS_CLASSES.index(worst_class)
        mask = (y_true == worst_idx)
        wrong_preds = p_cdan[mask][p_cdan[mask] != worst_idx]
        
        if len(wrong_preds) > 0:
            unique, counts = np.unique(wrong_preds, return_counts=True)
            dominant_wrong_idx = unique[np.argmax(counts)]
            print(f"Dominant confusion for {worst_class} in CDAN is: {PACS_CLASSES[dominant_wrong_idx]} ({np.max(counts)} times)")
        else:
            print(f"No confusions for {worst_class} in CDAN.")

if __name__ == "__main__":
    evaluate_all()