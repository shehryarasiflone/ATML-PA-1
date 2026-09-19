import os
import json
import zipfile
import urllib.request
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from common.seed import set_seed

PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]

def download_pacs(data_root: str = "./data"):
    """
    Downloads and extracts the official PACS dataset if not present.
    """
    root = Path(data_root)
    pacs_dir = root / "kfold"
    if pacs_dir.exists():
        return pacs_dir

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "pacs.zip"
    url = "https://www.dropbox.com/s/s7ptj047805x127/kfold.zip?dl=1"

    print("Downloading PACS dataset (~1.6 GB)...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting PACS dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root)

    if zip_path.exists():
        os.remove(zip_path)

    print(f"PACS dataset ready at {pacs_dir}")
    return pacs_dir

class PACSDataset(Dataset):
    """
    Custom Dataset loader for an indexed subset of PACS.
    """
    def __init__(self, samples, transform=None):
        """
        samples: list of tuples (img_path, class_idx, domain_idx)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, domain_idx

def generate_pacs_splits(data_root: str = "./data", output_dir: str = "./shared/splits"):
    """
    Generates stratified 80/20 train/val splits for Photo, Art, and Cartoon using seed 6304.
    Sketch is preserved in its entirety as the target evaluation/adaptation set.
    """
    set_seed(6304)
    pacs_root = download_pacs(data_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = {
        "dataset": "PACS",
        "seed": 6304,
        "classes": PACS_CLASSES,
        "domains": PACS_DOMAINS,
        "sources": {
            "art_painting": {"train": [], "val": []},
            "cartoon": {"train": [], "val": []},
            "photo": {"train": [], "val": []}
        },
        "target": {
            "sketch": []
        }
    }

    # Iterate over domains
    for domain_idx, domain in enumerate(PACS_DOMAINS):
        domain_folder = pacs_root / domain
        domain_samples = []

        for class_idx, cls_name in enumerate(PACS_CLASSES):
            cls_folder = domain_folder / cls_name
            if not cls_folder.exists():
                continue
            for img_file in sorted(os.listdir(cls_folder)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = str(cls_folder / img_file)
                    domain_samples.append((full_path, class_idx, domain_idx))

        if domain == "sketch":
            # Target domain: all images used for adaptation/eval
            split_manifest["target"]["sketch"] = domain_samples
            print(f"Target 'sketch': {len(domain_samples)} total images.")
        else:
            # Source domain: 80/20 stratified train/val split
            labels = [s[1] for s in domain_samples]
            train_items, val_items = train_test_split(
                domain_samples,
                test_size=0.20,
                stratify=labels,
                random_state=6304,
                shuffle=True
            )
            split_manifest["sources"][domain]["train"] = train_items
            split_manifest["sources"][domain]["val"] = val_items
            print(f"Source '{domain}': {len(train_items)} train, {len(val_items)} val images.")

    split_file = out_dir / "pacs_sketch_seed6304.json"
    with open(split_file, "w") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"Split manifest successfully saved to {split_file}")

if __name__ == "__main__":
    generate_pacs_splits()