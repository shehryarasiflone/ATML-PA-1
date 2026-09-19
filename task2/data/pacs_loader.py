import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from common.pacs import PACSDataset, PACS_CLASSES

# Preprocessing defined in the guide
TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

EVAL_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_pacs_dataloaders(split_path: str = "./shared/splits/pacs_sketch_seed6304.json"):
    with open(split_path, "r") as f:
        manifest = json.load(f)

    # 1. Source Train Loaders (8 samples per source domain per step = 24 total)
    source_train_loaders = {}
    for domain in ["photo", "art_painting", "cartoon"]:
        ds = PACSDataset(manifest["sources"][domain]["train"], transform=TRAIN_TRANSFORM)
        source_train_loaders[domain] = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

    # 2. Target Adaptation Loader (24 Sketch samples per step)
    target_adapt_ds = PACSDataset(manifest["target"]["sketch"], transform=TRAIN_TRANSFORM)
    target_adapt_loader = DataLoader(target_adapt_ds, batch_size=24, shuffle=True, drop_last=True)

    # 3. Source Validation Loaders
    source_val_loaders = {}
    for domain in ["photo", "art_painting", "cartoon"]:
        ds = PACSDataset(manifest["sources"][domain]["val"], transform=EVAL_TRANSFORM)
        source_val_loaders[domain] = DataLoader(ds, batch_size=64, shuffle=False)

    # 4. Final Target Evaluation Loader
    target_eval_ds = PACSDataset(manifest["target"]["sketch"], transform=EVAL_TRANSFORM)
    target_eval_loader = DataLoader(target_eval_ds, batch_size=64, shuffle=False)

    return source_train_loaders, target_adapt_loader, source_val_loaders, target_eval_loader