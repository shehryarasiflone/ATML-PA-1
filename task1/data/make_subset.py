import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10

from common.seed import set_seed

def generate_and_save_splits(data_root: str = "./data", output_dir: str = "./task1/data") -> None:
    set_seed(6304)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    #Load official partitions
    print("Loading official STL-10 dataset...")
    train_set = STL10(root=data_root, split="train", download=True)
    test_set = STL10(root=data_root, split="test", download=True)

    train_labels = np.array(train_set.labels)
    test_labels = np.array(test_set.labels)
    num_classes = len(np.unique(train_labels))

    #Stratified 80/20 train/val split from train
    train_indices = np.arange(len(train_labels))
    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.20,
        stratify=train_labels,
        random_state=6304,
        shuffle=True,
    )

    #Class-balanced 500-image evaluation subset from official test partition
    samples_per_class = 500 // num_classes
    rng = np.random.default_rng(6304)
    eval_subset_idx = []

    for c in range(num_classes):
        class_indices = np.where(test_labels == c)[0]
        selected = rng.choice(class_indices, size=samples_per_class, replace=False)
        eval_subset_idx.extend(selected.tolist())

    eval_subset_idx = sorted(eval_subset_idx)

    split_data = {
        "dataset": "STL-10",
        "seed": 6304,
        "classes": train_set.classes,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "eval_subset_indices": eval_subset_idx,
    }

    output_file = out_path / "splits_seed6304.json"
    with open(output_file, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"Splits saved successfully to {output_file}:")
    print(f" - Train samples: {len(train_idx)}")
    print(f" - Validation samples: {len(val_idx)}")
    print(f" - Evaluation subset samples: {len(eval_subset_idx)} (balanced 50/class)")

if __name__ == "__main__":
    generate_and_save_splits()