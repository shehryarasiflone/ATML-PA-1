import os
import math
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common.seed import set_seed
from task2.models.backbone import PACSResNet18, freeze_bn_stats
from task2.data.pacs_loader import get_pacs_dataloaders
from task2.methods.dann import DomainDiscriminator
from task2.scripts.train_source_only import evaluate_domain
from task2.scripts.evaluate_final import extract_features_and_preds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate_dann(max_alpha: float):
    set_seed(6304)
    src_train_loaders, target_adapt_loader, src_val_loaders, target_eval_loader = get_pacs_dataloaders()

    model = PACSResNet18(num_classes=7).to(DEVICE)
    discriminator = DomainDiscriminator(in_features=512, hidden_dim=256).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=1e-4, weight_decay=1e-4
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    num_batches = max(len(loader) for loader in src_train_loaders.values())
    total_epochs = 20  # Reduced slightly for study efficiency, early stopping usually triggers by 15
    total_steps = total_epochs * num_batches
    global_step = 0

    target_iter = iter(target_adapt_loader)
    best_val_f1 = -1.0
    patience = 5
    patience_counter = 0
    os.makedirs("task2/checkpoints/study", exist_ok=True)
    checkpoint_path = f"task2/checkpoints/study/dann_alpha_{max_alpha}.pth"

    print(f"\n--- Training DANN with max_alpha = {max_alpha} ---")

    for epoch in range(1, total_epochs + 1):
        model.train()
        discriminator.train()
        freeze_bn_stats(model)

        iter_loaders = {d: iter(loader) for d, loader in src_train_loaders.items()}

        for _ in range(num_batches):
            p = float(global_step) / float(total_steps)
            # Apply the max_alpha scaling to the standard DANN schedule
            alpha = max_alpha * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

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

            try:
                x_tgt, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_adapt_loader)
                x_tgt, _, _ = next(target_iter)
            x_tgt = x_tgt.to(DEVICE)

            optimizer.zero_grad()

            src_feat, src_logits = model(x_src)
            loss_cls = criterion_cls(src_logits, y_src)

            tgt_feat, _ = model(x_tgt)

            domain_feat = torch.cat([src_feat, tgt_feat], dim=0)
            domain_logits = discriminator(domain_feat, alpha).squeeze(-1)
            domain_labels = torch.cat([
                torch.zeros(x_src.size(0), device=DEVICE),
                torch.ones(x_tgt.size(0), device=DEVICE)
            ])

            loss_domain = criterion_domain(domain_logits, domain_labels)
            loss = loss_cls + loss_domain
            
            loss.backward()
            optimizer.step()
            global_step += 1

        # Validation
        val_f1s = []
        for d, loader in src_val_loaders.items():
            _, f1 = evaluate_domain(model, loader)
            val_f1s.append(f1)
        mean_val_f1 = np.mean(val_f1s)

        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluation Phase
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    tgt_acc, _ = evaluate_domain(model, target_eval_loader)

    # Calculate Domain Separability
    f_src, _, _ = extract_features_and_preds(model, src_val_loaders, is_source_dict=True)
    f_tgt, _, _ = extract_features_and_preds(model, tgt_eval_loader, is_source_dict=False)
    
    min_samples = min(len(f_src), len(f_tgt))
    idx_src = np.random.choice(len(f_src), min_samples, replace=False)
    idx_tgt = np.random.choice(len(f_tgt), min_samples, replace=False)
    
    X_domain = np.vstack([f_src[idx_src], f_tgt[idx_tgt]])
    y_domain = np.concatenate([np.zeros(min_samples), np.ones(min_samples)])
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_domain, y_domain, test_size=0.3, random_state=6304, stratify=y_domain)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    sep_score = clf.score(X_te, y_te) * 100

    return best_val_f1, tgt_acc, sep_score

def run_study():
    alphas = [0.25, 0.5, 1.0]
    results = []

    for a in alphas:
        src_f1, tgt_acc, sep = train_and_evaluate_dann(a)
        results.append((a, src_f1, tgt_acc, sep))

    print("\n" + "="*80)
    print("DANN Controlled Design Study: Varying max_alpha")
    print("="*80)
    print(f"{'max_alpha':<12} | {'Source Val F1':<15} | {'Target Acc':<15} | {'Domain Separability'}")
    print("-" * 80)
    for a, src_f1, tgt_acc, sep in results:
        print(f"{a:<12.2f} | {src_f1:>14.2f}% | {tgt_acc:>14.2f}% | {sep:>18.2f}%")
    print("="*80)

if __name__ == "__main__":
    run_study()