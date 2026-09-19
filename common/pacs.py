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

def locate_pacs_root(search_roots=None):
    """
    Scans candidate directories (including Kaggle input) for the PACS domains.
    """
    if search_roots is None:
        search_roots = [
            "/kaggle/input",
            "./data",
            "./data/kfold",
            "./data/PACS",
            "../data"
        ]

    for candidate in search_roots:
        cand_path = Path(candidate)
        if not cand_path.exists():
            continue

        # Check direct child directories
        for root, dirs, _ in os.walk(cand_path):
            dir_names = set(d.lower() for d in dirs)
            if all(dom in dir_names for dom in PACS_DOMAINS):
                print(f"Discovered PACS root at: {root}")
                return Path(root)

    return None

def download_pacs(data_root: str = "./data"):
    """
    Downloads and extracts PACS if not already found in local paths or Kaggle input.
    """
    # 1. First check if PACS is already mounted in Kaggle input or local disk
    existing_root = locate_pacs_root()
    if existing_root is not None:
        return existing_root

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "pacs_kfold.zip"

    # Remove any corrupted prior download
    if zip_path.exists():
        os.remove(zip_path)

    # Direct mirror URL with browser User-Agent
    url = "https://dl.dropboxusercontent.com/s/s7ptj047805x127/kfold.zip"
    print("Downloading PACS dataset (~1.6 GB) from direct mirror...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        chunk_size = 16 * 1024 * 1024  # 16 MB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)

    print("Extracting PACS dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root)

    if zip_path.exists():
        os.remove(zip_path)

    found_root = locate_pacs_root([root])
    if found_root is None:
        raise FileNotFoundError("Extracted archive did not match expected PACS folder structure.")

    return found_root

class PACSDataset(Dataset):
    """
    Custom Dataset loader for an indexed subset of PACS.
    """
    def __init__(self, samples, transform=None):
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
    Sketch is preserved entirely for the target adaptation/evaluation set.
    """
    set_seed(6304)
    pacs_root = locate_pacs_root()
    if pacs_root is None:
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

    # Inspect each domain folder
    for domain_idx, domain in enumerate(PACS_DOMAINS):
        # Case-insensitive domain lookup
        matching_dirs = [d for d in os.listdir(pacs_root) if d.lower() == domain]
        if not matching_dirs:
            raise FileNotFoundError(f"Domain folder '{domain}' missing in {pacs_root}")
        domain_folder = pacs_root / matching_dirs[0]

        domain_samples = []
        for class_idx, cls_name in enumerate(PACS_CLASSES):
            cls_matches = [c for c in os.listdir(domain_folder) if c.lower() == cls_name]
            if not cls_matches:
                continue
            cls_folder = domain_folder / cls_matches[0]
            for img_file in sorted(os.listdir(cls_folder)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = str(cls_folder / img_file)
                    domain_samples.append((full_path, class_idx, domain_idx))

        if domain == "sketch":
            split_manifest["target"]["sketch"] = domain_samples
            print(f"Target 'sketch': {len(domain_samples)} total images.")
        else:
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

    print(f"\nSplit manifest successfully generated and saved to {split_file}")

if __name__ == "__main__":
    generate_pacs_splits()