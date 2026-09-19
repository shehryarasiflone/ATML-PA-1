import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from common.seed import set_seed
from task2.models.backbone import PACSResNet18, freeze_bn_stats
from task2.data.pacs_loader import get_pacs_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_domain(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE)
        _, logits = model(imgs)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)
    acc = np.mean(preds_np == targets_np) * 100.0
    macro_f1 = f1_score(targets_np, preds_np, average="macro") * 100.0
    return acc, macro_f1

def train_source_only():
    set_seed(6304)
    src_train_loaders, _, src_val_loaders, target_eval_loader = get_pacs_dataloaders()

    model = PACSResNet18(num_classes=7).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Determine number of batches per epoch from largest source loader
    num_batches = max(len(loader) for loader in src_train_loaders.values())

    best_val_f1 = -1.0
    patience = 5
    patience_counter = 0
    os.makedirs("task2/checkpoints", exist_ok=True)
    best_checkpoint_path = "task2/checkpoints/source_only_best.pth"

    print("Training Source-Only ERM (Photo, Art, Cartoon) with frozen BatchNorm stats...")

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

            # Combined balanced source batch: 24 samples
            x_src = torch.cat(batch_imgs, dim=0).to(DEVICE)
            y_src = torch.cat(batch_labels, dim=0).to(DEVICE)

            optimizer.zero_grad()
            _, logits = model(x_src)
            loss = criterion(logits, y_src)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation across all 3 source domains
        val_accs, val_f1s = {}, {}
        for d, loader in src_val_loaders.items():
            acc, f1 = evaluate_domain(model, loader)
            val_accs[d], val_f1s[d] = acc, f1

        mean_val_acc = np.mean(list(val_accs.values()))
        mean_val_f1 = np.mean(list(val_f1s.values()))

        print(f"Epoch {epoch:02d} | Loss: {epoch_loss/num_batches:.4f} | "
              f"Source Val Acc: {mean_val_acc:.2f}% | Source Val Macro-F1: {mean_val_f1:.2f}%")

        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            torch.save(model.state_dict(), best_checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Load best checkpoint for reporting
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))

    # Evaluate on individual source validation domains
    print("\n" + "="*70)
    print(f"{'Domain':<20} | {'Validation Acc (%)':<20} | {'Macro-F1 (%)':<20}")
    print("="*70)
    for d, loader in src_val_loaders.items():
        acc, f1 = evaluate_domain(model, loader)
        print(f"{d:<20} | {acc:<20.2f} | {f1:<20.2f}")

    # Evaluate on unseen target (Sketch)
    tgt_acc, tgt_f1 = evaluate_domain(model, target_eval_loader)
    print("-"*70)
    print(f"{'Target (Sketch)':<20} | {tgt_acc:<20.2f} | {tgt_f1:<20.2f}")
    print("="*70)

if __name__ == "__main__":
    train_source_only()