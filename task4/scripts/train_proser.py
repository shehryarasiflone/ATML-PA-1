import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.seed import set_seed
from task4.models.resnet_cifar import ResNet18CIFAR
from task4.data.cifar_loaders import get_osr_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_csa(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        _, logits = model(imgs)
        # Closed-set classification uses only the 10 known-class logits
        preds = torch.argmax(logits[:, :10], dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return (correct / total) * 100.0

def train_proser():
    set_seed(6304)
    train_loader, val_loader, test_loader, _, _ = get_osr_dataloaders()

    # 1. Initialize ResNet-18 with 15 classes (10 known + 5 dummy placeholders)
    model = ResNet18CIFAR(num_classes=15).to(DEVICE)

    # 2. Load weights from Vanilla checkpoint for backbone and first 10 class logits
    vanilla_path = "task4/checkpoints/vanilla_best.pth"
    assert os.path.exists(vanilla_path), f"Missing {vanilla_path}. Train vanilla first!"
    vanilla_state = torch.load(vanilla_path, map_location=DEVICE)

    model_state = model.state_dict()
    for k, v in vanilla_state.items():
        if k == "fc.weight":
            model_state["fc.weight"][:10, :] = v
        elif k == "fc.bias":
            model_state["fc.bias"][:10] = v
        else:
            model_state[k] = v
    model.load_state_dict(model_state)

    # 3. Optimization budget: 50 epochs, SGD lr=1e-3, momentum=0.9, weight_decay=5e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    beta_dist = torch.distributions.Beta(2.0, 2.0)
    best_val_acc = -1.0
    os.makedirs("task4/checkpoints", exist_ok=True)
    checkpoint_path = "task4/checkpoints/proser_best.pth"

    print("Training PROSER (Classifier + Data Placeholders, 50 epochs, lr=1e-3)...")

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            b = imgs.size(0)
            mid = b // 2

            # Split mini-batch into two equal halves
            x_cp, y_cp = imgs[:mid], labels[:mid]
            x_dp, y_dp = imgs[mid:], labels[mid:]

            # --- Part 1: Classifier Placeholders (First Half) ---
            h_cp = model.forward_features_stage1(x_cp)
            feat_cp = model.forward_features_stage2(h_cp)
            logits_cp = model.fc(feat_cp)

            # Known-class classification loss
            loss_cls = F.cross_entropy(logits_cp[:, :10], y_cp)

            # Classifier placeholder loss (beta = 1.0):
            # Target is the strongest dummy classifier (classes 10..14)
            dummy_logits_cp = logits_cp[:, 10:]
            target_cp = 10 + torch.argmax(dummy_logits_cp, dim=-1)

            # Mask out the ground-truth known class
            logits_masked = logits_cp.clone()
            logits_masked.scatter_(1, y_cp.unsqueeze(1), -1e9)
            loss_cp = F.cross_entropy(logits_masked, target_cp)

            # --- Part 2: Manifold-Mixup Data Placeholders (Second Half) ---
            # Generate pairs with y_i != y_j
            n_dp = x_dp.size(0)
            perm = torch.randperm(n_dp, device=DEVICE)
            for i in range(n_dp):
                if y_dp[i] == y_dp[perm[i]]:
                    diff_idx = (y_dp != y_dp[i]).nonzero(as_tuple=True)[0]
                    if len(diff_idx) > 0:
                        perm[i] = diff_idx[torch.randint(len(diff_idx), (1,)).item()]

            h_dp = model.forward_features_stage1(x_dp)
            lam = beta_dist.sample((n_dp, 1, 1, 1)).to(DEVICE)
            h_mixed = lam * h_dp + (1.0 - lam) * h_dp[perm]

            feat_mixed = model.forward_features_stage2(h_mixed)
            logits_mixed = model.fc(feat_mixed)

            # Synthetic mixup samples are trained toward dummy classifiers (gamma = 0.1)
            target_dp = 10 + torch.argmax(logits_mixed[:, 10:], dim=-1)
            loss_dp = F.cross_entropy(logits_mixed, target_dp)

            # Combined PROSER objective
            loss = loss_cls + 1.0 * loss_cp + 0.1 * loss_dp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 5 == 0 or epoch == 50:
            val_acc = evaluate_csa(model, val_loader)
            print(f"Epoch {epoch:02d}/50 | Loss: {total_loss/len(train_loader):.4f} | CIFAR-10 Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_acc = evaluate_csa(model, test_loader)
    print("\n" + "="*50)
    print(f"PROSER Best Val Acc  : {best_val_acc:.2f}%")
    print(f"PROSER Test Acc (CSA): {test_acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    train_proser()