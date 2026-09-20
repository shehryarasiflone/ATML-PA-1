import os
import torch
import torch.nn as nn
import numpy as np

from common.seed import set_seed
from task2.models.backbone import PACSResNet18, freeze_bn_stats
from task2.data.pacs_loader import get_pacs_dataloaders
from task2.scripts.train_source_only import evaluate_domain
from task3.methods.dan_dg import PairwiseSourceMMD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dan_dg(lambda_dg: float = 1.0):
    set_seed(6304)
    # Notice: target_adapt_loader and target_eval_loader are intentionally omitted from training
    src_train_loaders, _, src_val_loaders, _ = get_pacs_dataloaders()

    model = PACSResNet18(num_classes=7).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pairwise_mmd = PairwiseSourceMMD()

    num_batches = max(len(loader) for loader in src_train_loaders.values())
    best_val_f1 = -1.0
    patience = 5
    patience_counter = 0

    os.makedirs("task3/checkpoints", exist_ok=True)
    checkpoint_path = f"task3/checkpoints/dan_dg_lambda_{lambda_dg}_best.pth"

    print(f"Training DAN-DG (λ_DG = {lambda_dg}) on Photo, Art, Cartoon with frozen BatchNorm...")

    for epoch in range(1, 31):
        model.train()
        freeze_bn_stats(model)

        iter_loaders = {d: iter(loader) for d, loader in src_train_loaders.items()}
        epoch_cls_loss, epoch_mmd_loss = 0.0, 0.0

        for _ in range(num_batches):
            feats_by_domain = {}
            logits_list, labels_list = [], []

            for d, loader in src_train_loaders.items():
                try:
                    imgs, labels, _ = next(iter_loaders[d])
                except StopIteration:
                    iter_loaders[d] = iter(loader)
                    imgs, labels, _ = next(iter_loaders[d])

                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                feat, logits = model(imgs)

                feats_by_domain[d] = feat
                logits_list.append(logits)
                labels_list.append(labels)

            # Combined source classification loss
            all_logits = torch.cat(logits_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            loss_cls = criterion_cls(all_logits, all_labels)

            # Pairwise source MMD discrepancy
            loss_mmd = criterion_pairwise_mmd(feats_by_domain)

            loss = loss_cls + lambda_dg * loss_mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_cls_loss += loss_cls.item()
            epoch_mmd_loss += loss_mmd.item()

        # Validation across source domains
        val_accs, val_f1s = {}, {}
        for d, loader in src_val_loaders.items():
            acc, f1 = evaluate_domain(model, loader)
            val_accs[d], val_f1s[d] = acc, f1

        mean_val_acc = np.mean(list(val_accs.values()))
        mean_val_f1 = np.mean(list(val_f1s.values()))
        worst_val_f1 = min(val_f1s.values())
        worst_domain = min(val_f1s, key=val_f1s.get)

        print(f"Epoch {epoch:02d} | Cls Loss: {epoch_cls_loss/num_batches:.4f} | "
              f"Pairwise MMD: {epoch_mmd_loss/num_batches:.4f} | "
              f"Mean Val F1: {mean_val_f1:.2f}% | Worst ({worst_domain}): {worst_val_f1:.2f}%")

        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train_dan_dg(lambda_dg=1.0)