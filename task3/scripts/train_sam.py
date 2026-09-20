import os
import torch
import torch.nn as nn
import numpy as np

from common.seed import set_seed
from task2.models.backbone import PACSResNet18, freeze_bn_stats
from task2.data.pacs_loader import get_pacs_dataloaders
from task2.scripts.train_source_only import evaluate_domain
from task3.methods.sam import SAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_sam(rho: float = 0.05):
    set_seed(6304)
    # Target Sketch is intentionally excluded from training and checkpoint selection
    src_train_loaders, _, src_val_loaders, _ = get_pacs_dataloaders()

    model = PACSResNet18(num_classes=7).to(DEVICE)
    
    # Initialize SAM wrapping AdamW with identical lr=1e-4 and weight_decay=1e-4
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

    os.makedirs("task3/checkpoints", exist_ok=True)
    checkpoint_path = f"task3/checkpoints/sam_rho_{rho}_best.pth"

    print(f"Training SAM (ρ = {rho}) on Photo, Art, Cartoon with frozen BatchNorm stats...")

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

            # --- SAM Pass 1: Compute gradient at theta to find worst-case epsilon ---
            freeze_bn_stats(model)
            _, logits = model(x_src)
            loss1 = criterion(logits, y_src)
            loss1.backward()
            optimizer.first_step(zero_grad=True)

            # --- SAM Pass 2: Compute gradient at theta + epsilon and update theta ---
            freeze_bn_stats(model)
            _, logits_perturbed = model(x_src)
            loss2 = criterion(logits_perturbed, y_src)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            epoch_loss += loss1.item()

        # Validation across source domains
        val_accs, val_f1s = {}, {}
        for d, loader in src_val_loaders.items():
            acc, f1 = evaluate_domain(model, loader)
            val_accs[d], val_f1s[d] = acc, f1

        mean_val_acc = np.mean(list(val_accs.values()))
        mean_val_f1 = np.mean(list(val_f1s.values()))
        worst_val_f1 = min(val_f1s.values())
        worst_domain = min(val_f1s, key=val_f1s.get)

        print(f"Epoch {epoch:02d} | Loss: {epoch_loss/num_batches:.4f} | "
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
    train_sam(rho=0.05)