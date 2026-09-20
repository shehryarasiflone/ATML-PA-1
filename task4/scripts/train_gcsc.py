import os
import torch
import torch.nn as nn
from common.seed import set_seed
from task4.models.resnet_cifar import ResNet18CIFAR
from task4.data.cifar_loaders import get_osr_dataloaders
from task4.scripts.train_vanilla import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_gcsc():
    set_seed(6304)
    # Load training data with RandAugment (num_ops=2, magnitude=9)
    train_loader, val_loader, test_loader, _, _ = get_osr_dataloaders(use_randaugment=True)

    model = ResNet18CIFAR(num_classes=10).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    os.makedirs("task4/checkpoints", exist_ok=True)
    checkpoint_path = "task4/checkpoints/gcsc_best.pth"

    print("Training GCSC ResNet-18 with RandAugment (100 epochs, Cosine Annealing)...")

    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            _, logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0 or epoch == 100:
            val_acc = evaluate(model, val_loader)
            print(f"Epoch {epoch:03d}/100 | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)

    # Evaluate best checkpoint on CIFAR-10 test set
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_acc = evaluate(model, test_loader)
    print("\n" + "="*50)
    print(f"GCSC Best Val Acc  : {best_val_acc:.2f}%")
    print(f"GCSC Test Acc (CSA): {test_acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    train_gcsc()