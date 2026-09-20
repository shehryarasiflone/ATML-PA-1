import json
import os
from pathlib import Path
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.model_selection import train_test_split

from common.seed import set_seed

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]

def locate_or_link_cifar(is_cifar100: bool = False, local_data_root: str = "./data"):
    """
    Finds the dataset in /kaggle/input or local directories.
    If the internal folder structure does not match torchvision's expected
    'cifar-10-batches-py' or 'cifar-100-python', creates a symlink in local_data_root
    so torchvision loads the Kaggle files instantly with download=False.
    """
    target_folder = "cifar-100-python" if is_cifar100 else "cifar-10-batches-py"
    marker_file = "meta" if is_cifar100 else "batches.meta"
    
    local_root = Path(local_data_root).resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    expected_local_dir = local_root / target_folder

    # If already set up locally, return
    if expected_local_dir.exists():
        return str(local_root)

    # Search candidates
    search_dirs = ["/kaggle/input", "./data", "../data", "/kaggle/working/ATML-PA-1/data"]
    
    # 1. First search for the exact folder name
    for s_dir in search_dirs:
        p = Path(s_dir)
        if not p.exists():
            continue
        for root, dirs, _ in os.walk(p):
            if target_folder in dirs:
                print(f"Discovered standard folder '{target_folder}' at: {root}")
                return str(root)

    # 2. Search for the marker file directly if folder was renamed
    print(f"Searching for raw {marker_file} for {'CIFAR-100' if is_cifar100 else 'CIFAR-10'}...")
    for s_dir in search_dirs:
        p = Path(s_dir)
        if not p.exists():
            continue
        for root, _, files in os.walk(p):
            # Check for CIFAR-100 or CIFAR-10 marker
            if marker_file in files:
                # Differentiate between CIFAR-10 and CIFAR-100
                if is_cifar100 and ("100" in root.lower() or "meta" in files):
                    print(f"Found CIFAR-100 files at: {root}")
                    # Create symlink so torchvision finds ./data/cifar-100-python
                    if not expected_local_dir.exists():
                        os.symlink(root, expected_local_dir)
                        print(f"Created symlink: {expected_local_dir} -> {root}")
                    return str(local_root)
                elif not is_cifar100 and ("10" in root.lower() or "batches.meta" in files):
                    print(f"Found CIFAR-10 files at: {root}")
                    if not expected_local_dir.exists():
                        os.symlink(root, expected_local_dir)
                    return str(local_root)

    raise FileNotFoundError(f"Could not locate {'CIFAR-100' if is_cifar100 else 'CIFAR-10'} in inputs.")

def prepare_osr_splits(output_dir: str = "task4/splits"):
    set_seed(6304)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c10_root = locate_or_link_cifar(is_cifar100=False)
    c100_root = locate_or_link_cifar(is_cifar100=True)

    print(f"Loading CIFAR-10 from: {c10_root}")
    c10_train = CIFAR10(root=c10_root, train=True, download=False)
    c10_test = CIFAR10(root=c10_root, train=False, download=False)

    train_indices, val_indices = train_test_split(
        np.arange(len(c10_train)),
        test_size=0.10,
        stratify=c10_train.targets,
        random_state=6304,
        shuffle=True
    )

    print(f"CIFAR-10 Splits -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(c10_test)}")

    print(f"Loading CIFAR-100 from: {c100_root}")
    c100_test = CIFAR100(root=c100_root, train=False, download=False)
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

    print(f"Manifest successfully generated and saved to {split_path}")

if __name__ == "__main__":
    prepare_osr_splits()