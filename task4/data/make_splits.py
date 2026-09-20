import json
import os
from pathlib import Path
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.model_selection import train_test_split

from common.seed import set_seed

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]

def prepare_osr_splits(data_root: str = "./data", output_dir: str = "task4/splits"):
    set_seed(6304)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load CIFAR-10
    c10_train = CIFAR10(root=data_root, train=True, download=True)
    c10_test = CIFAR10(root=data_root, train=False, download=True)

    train_indices, val_indices = train_test_split(
        np.arange(len(c10_train)),
        test_size=0.10,
        stratify=c10_train.targets,
        random_state=6304,
        shuffle=True
    )

    print(f"CIFAR-10 Splits -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(c10_test)}")

    # 2. Load CIFAR-100 Test Set for Unknowns
    c100_test = CIFAR100(root=data_root, train=False, download=True)
    c100_classes = c100_test.classes

    near_indices = []
    far_indices = []

    for idx, (_, target_idx) in enumerate(c100_test):
        class_name = c100_classes[target_idx]
        if class_name in NEAR_CLASSES:
            near_indices.append(idx)
        elif class_name in FAR_CLASSES:
            far_indices.append(idx)

    print(f"CIFAR-100 Unknowns -> Near: {len(near_indices)} images, Far: {len(far_indices)} images")
    assert len(near_indices) == 800, f"Expected 800 near unknowns, got {len(near_indices)}"
    assert len(far_indices) == 800, f"Expected 800 far unknowns, got {len(far_indices)}"

    split_manifest = {
        "dataset": "CIFAR-10 / CIFAR-100 OSR",
        "seed": 6304,
        "cifar10_classes": c10_train.classes,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "near_classes": NEAR_CLASSES,
        "near_indices": near_indices,
        "far_classes": FAR_CLASSES,
        "far_indices": far_indices
    }

    split_path = out_dir / "cifar_osr_seed6304.json"
    with open(split_path, "w") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"Manifest successfully written to {split_path}")

if __name__ == "__main__":
    prepare_osr_splits()