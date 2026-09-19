import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.seed import set_seed
from task2.models.backbone import PACSResNet18, freeze_bn_stats
from task2.data.pacs_loader import get_pacs_dataloaders
from task2.methods.cdan import ConditionalDomainDiscriminator, calc_entropy_weights
from task2.scripts.train_source_only import evaluate_domain

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_cdan():
    set_seed(6304)
    src_train_loaders, target_adapt_loader, src_val_loaders, target_eval_loader = get_pacs_dataloaders()

    model = PACSResNet18(num_classes=7).to(DEVICE)
    # Multilinear dimension: 512 features * 7 classes = 3584
    discriminator = ConditionalDomainDiscriminator(in_features=3584, hidden_dim=1024).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss(reduction='none')

    num_batches = max(len(loader) for loader in src_train_loaders.values())
    total_epochs = 30
    total_steps = total_epochs * num_batches
    global_step = 0

    target_iter = iter(target_adapt_loader)
    best_val_f1 = -1.0
    patience = 5
    patience_counter = 0
    os.makedirs("task2/checkpoints", exist_ok=True)
    best_checkpoint_path = "task2/checkpoints/cdan_best.pth"

    print("Training CDAN+E (Conditional GRL Alignment) with frozen BatchNorm stats...")

    for epoch in range(1, total_epochs + 1):
        model.train()
        discriminator.train()
        freeze_bn_stats(model)

        iter_loaders = {d: iter(loader) for d, loader in src_train_loaders.items()}
        epoch_cls_loss, epoch_domain_loss = 0.0, 0.0

        for _ in range(num_batches):
            p = float(global_step) / float(total_steps)
            alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

            # 1. Balanced source mini-batch (24 samples)
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

            # 2. Target mini-batch (24 Sketch samples)
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
            src_probs = F.softmax(src_logits, dim=-1)

            # Forward pass target
            tgt_feat, tgt_logits = model(x_tgt)
            tgt_probs = F.softmax(tgt_logits, dim=-1)

            # Combine conditioned features
            joint_feat = torch.cat([src_feat, tgt_feat], dim=0)
            joint_probs = torch.cat([src_probs, tgt_probs], dim=0)

            # Domain prediction
            domain_logits = discriminator(joint_feat, joint_probs, alpha).squeeze(-1)
            domain_labels = torch.cat([
                torch.zeros(x_src.size(0), device=DEVICE),
                torch.ones(x_tgt.size(0), device=DEVICE)
            ])

            # Entropy weighting (CDAN+E)
            sample_weights = calc_entropy_weights(joint_probs)
            raw_domain_loss = criterion_domain(domain_logits, domain_labels)
            loss_domain = torch.mean(sample_weights * raw_domain_loss)

            loss = loss_cls + loss_domain
            loss.backward()
            optimizer.step()

            epoch_cls_loss += loss_cls.item()
            epoch_domain_loss += loss_domain.item()
            global_step += 1

        # Checkpoint selection on source validation
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

    # Final evaluation on target Sketch
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
    tgt_acc, tgt_f1 = evaluate_domain(model, target_eval_loader)
    print("\n" + "="*50)
    print(f"CDAN Final Target (Sketch) Accuracy : {tgt_acc:.2f}%")
    print(f"CDAN Final Target (Sketch) Macro-F1 : {tgt_f1:.2f}%")
    print("="*50)

if __name__ == "__main__":
    train_cdan()