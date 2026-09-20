import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as T

CIFAR_TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

CIFAR_EVAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def get_osr_dataloaders(split_path: str = "task4/splits/cifar_osr_seed6304.json"):
    with open(split_path, "r") as f:
        manifest = json.load(f)

    c10_root = manifest.get("c10_root", "./data")
    c100_root = manifest.get("c100_root", "./data")

    # Base datasets read directly from discovered root with download=False
    c10_train_raw = CIFAR10(root=c10_root, train=True, download=False, transform=CIFAR_TRAIN_TRANSFORM)
    c10_val_raw = CIFAR10(root=c10_root, train=True, download=False, transform=CIFAR_EVAL_TRANSFORM)
    c10_test = CIFAR10(root=c10_root, train=False, download=False, transform=CIFAR_EVAL_TRANSFORM)
    c100_test = CIFAR100(root=c100_root, train=False, download=False, transform=CIFAR_EVAL_TRANSFORM)

    # Subsets
    train_ds = Subset(c10_train_raw, manifest["train_indices"])
    val_ds = Subset(c10_val_raw, manifest["val_indices"])
    near_ds = Subset(c100_test, manifest["near_indices"])
    far_ds = Subset(c100_test, manifest["far_indices"])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(c10_test, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    near_loader = DataLoader(near_ds, batch_size=128, shuffle=False, num_workers=2)
    far_loader = DataLoader(far_ds, batch_size=128, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, near_loader, far_loader