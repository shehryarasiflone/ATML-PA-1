import os
import math
import torch
import torch.nn as nn
import numpy as np

from common.seed import set_seed
from task2.models.backbone import PACSResNet18, freeze_bn_stats
from task2.data.pacs_loader import get_pacs_dataloaders
from task2.methods.dann import DomainDiscriminator
from task2.scripts.train_source_only import evaluate_domain

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dann():
    set_seed(6304)
    src_train_loaders, target_adapt_loader, src_val_loaders, target_eval_loader = get_pacs_dataloaders()

    model = PACSResNet18(num_classes=7).to(DEVICE)
    discriminator = DomainDiscriminator(in_features=512, hidden_dim=256).to(DEVICE)

    # Joint optimizer over backbone + classifier + discriminator
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    num_batches = max(len(loader) for loader in src_train_loaders.values())
    total_epochs = 30
    total_steps = total_epochs * num_batches
    global_step = 0

    target_iter = iter(target_adapt_loader)
    best_val_f1 = -1.0
    patience = 5
    patience_counter = 0
    os.makedirs("task2/checkpoints", exist_ok=True)
    best_checkpoint_path = "task2/checkpoints/dann_best.pth"

    print("Training DANN (Adversarial GRL) with frozen BatchNorm stats...")

    for epoch in range(1, total_epochs + 1):
        model.train()
        discriminator.train()
        freeze_bn_stats(model)

        iter_loaders = {d: iter(loader) for d, loader in src_train_loaders.items()}
        epoch_cls_loss, epoch_domain_loss = 0.0, 0.0

        for _ in range(num_batches):
            # Dynamic GRL progress schedule: p in [0, 1], lambda_p from 0 to 1
            p = float(global_step) / float(total_steps)
            alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

            # 1. Fetch balanced source mini-batch (24 samples)
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

            # 2. Fetch target mini-batch (24 Sketch samples)
            try:
                x_tgt, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_adapt_loader)
                x_tgt, _, _ = next(target_iter)
            x_tgt = x_tgt.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass source
            src_feat, src_logits = model(x_src)
            loss_cls = criterion_cls(src_logits, y_src)

            # Forward pass target
            tgt_feat, _ = model(x_tgt)

            # Combine features for domain discrimination
            # Source label = 0.0, Target label = 1.0
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

            epoch_cls_loss += loss_cls.item()
            epoch_domain_loss += loss_domain.item()
            global_step += 1

        # Validation on source domains (checkpoint selection)
        val_f1s = []
        for d, loader in src_val_loaders.items():
            _, f1 = evaluate_domain(model, loader)
            val_f1s.append(f1)
        mean_val_f1 = np.mean(val_f1s)

        print(f"Epoch {epoch:02d} | Cls Loss: {epoch_cls_loss/num_batches:.4f} | "
              f"Domain Loss: {epoch_domain_loss/num_batches:.4f} | α: {alpha:.3f} | "
              f"Source Val Macro-F1: {mean_val_f1:.2f}%")

        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            torch.save(model.state_dict(), best_checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Load best checkpoint and evaluate on target Sketch
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
    tgt_acc, tgt_f1 = evaluate_domain(model, target_eval_loader)
    print("\n" + "="*50)
    print(f"DANN Final Target (Sketch) Accuracy : {tgt_acc:.2f}%")
    print(f"DANN Final Target (Sketch) Macro-F1 : {tgt_f1:.2f}%")
    print("="*50)

if __name__ == "__main__":
    train_dann()