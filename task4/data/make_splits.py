import json
import os
from pathlib import Path
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.model_selection import train_test_split

from common.seed import set_seed

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]

def find_cifar_root(base_folder_name, search_dirs=None):
    """
    Finds the directory containing the CIFAR batch folder in Kaggle input or local storage.
    """
    if search_dirs is None:
        search_dirs = ["/kaggle/input", "./data", "../data", "/kaggle/working/ATML-PA-1/data"]
    for s_dir in search_dirs:
        p = Path(s_dir)
        if not p.exists():
            continue
        for root, dirs, _ in os.walk(p):
            if base_folder_name in dirs:
                print(f"Discovered {base_folder_name} at: {root}")
                return Path(root)
    return None

def prepare_osr_splits(output_dir: str = "task4/splits"):
    set_seed(6304)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate or fallback CIFAR-10
    c10_root = find_cifar_root("cifar-10-batches-py")
    if c10_root is None:
        c10_root = Path("./data")
        download_c10 = True
    else:
        download_c10 = False

    # 2. Locate or fallback CIFAR-100
    c100_root = find_cifar_root("cifar-100-python")
    if c100_root is None:
        c100_root = Path("./data")
        download_c100 = True
    else:
        download_c100 = False

    print(f"Loading CIFAR-10 from: {c10_root}")
    c10_train = CIFAR10(root=str(c10_root), train=True, download=download_c10)
    c10_test = CIFAR10(root=str(c10_root), train=False, download=download_c10)

    train_indices, val_indices = train_test_split(
        np.arange(len(c10_train)),
        test_size=0.10,
        stratify=c10_train.targets,
        random_state=6304,
        shuffle=True
    )

    print(f"CIFAR-10 Splits -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(c10_test)}")

    print(f"Loading CIFAR-100 from: {c100_root}")
    c100_test = CIFAR100(root=str(c100_root), train=False, download=download_c100)
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
        "c10_root": str(c10_root),
        "c100_root": str(c100_root),
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