import json
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset
from PIL import Image
import open_clip

from common.seed import set_seed
from task1.models.backbones import ResNet50FeatureExtractor, ViTB16FeatureExtractor, CLIPFeatureExtractor
from task1.models.classifier_head import LinearProbe
from task1.data.transforms import apply_patch_shuffle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_predictions(extractor, head, pil_images, cfg, is_clip_zs=False, class_names=None):
    extractor.eval()
    if head is not None:
        head.eval()

    norm_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    if is_clip_zs:
        clip_model = extractor.model
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        prompts = [f"a photo of a {c}." for c in class_names]
        text_tokens = tokenizer(prompts).to(DEVICE)
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        logit_scale = clip_model.logit_scale.exp()

    all_preds = []
    batch_size = 64
    for i in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[i:i + batch_size]
        tensors = torch.stack([norm_transform(img) for img in batch_imgs]).to(DEVICE)
        feats = extractor(tensors)
        if is_clip_zs:
            logits = logit_scale * (feats @ text_feats.T)
        else:
            logits = head(feats)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)

    return np.array(all_preds)

def evaluate_patch_shuffling():
    set_seed(6304)
    with open("task1/data/splits_seed6304.json", "r") as f:
        splits = json.load(f)

    classes = splits["classes"]
    eval_indices = splits["eval_subset_indices"]
    raw_test = STL10(root="./data", split="test", download=False)

    # 1. Prepare Clean and Deterministically Shuffled Image Sets
    clean_images = [raw_test[idx][0].resize((224, 224)) for idx in eval_indices]
    targets = np.array([raw_test[idx][1] for idx in eval_indices])

    # Distinct deterministic seed per image derived from base seed 6304
    shuffled_images = [
        apply_patch_shuffle(img, seed=6304 + i) for i, img in enumerate(clean_images)
    ]

    models_config = {
        "ResNet-50 (Linear Head)": {
            "extractor": ResNet50FeatureExtractor().to(DEVICE),
            "head_path": "task1/models/ResNet-50_head.pth",
            "dim": 2048,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "is_zs": False,
        },
        "ViT-B/16 (Linear Head)": {
            "extractor": ViTB16FeatureExtractor().to(DEVICE),
            "head_path": "task1/models/ViT-B_16_head.pth",
            "dim": 768,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "is_zs": False,
        },
        "CLIP (Linear Head)": {
            "extractor": CLIPFeatureExtractor().to(DEVICE),
            "head_path": "task1/models/CLIP (ViT-B_32)_head.pth",
            "dim": 512,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
            "is_zs": False,
        },
        "CLIP (Zero-Shot)": {
            "extractor": CLIPFeatureExtractor().to(DEVICE),
            "head_path": None,
            "dim": 512,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
            "is_zs": True,
        },
    }

    results = []

    for model_name, cfg in models_config.items():
        print(f"Evaluating {model_name} on patch shuffling...")
        head = None
        if not cfg["is_zs"]:
            head = LinearProbe(cfg["dim"]).to(DEVICE)
            head.load_state_dict(torch.load(cfg["head_path"], map_location=DEVICE))

        # Evaluate clean images
        clean_preds = get_predictions(cfg["extractor"], head, clean_images, cfg, is_clip_zs=cfg["is_zs"], class_names=classes)
        clean_acc = np.mean(clean_preds == targets) * 100.0

        # Evaluate patch-shuffled images
        shuffled_preds = get_predictions(cfg["extractor"], head, shuffled_images, cfg, is_clip_zs=cfg["is_zs"], class_names=classes)
        shuffled_acc = np.mean(shuffled_preds == targets) * 100.0

        delta_acc = shuffled_acc - clean_acc
        consistency = np.mean(shuffled_preds == clean_preds) * 100.0

        results.append({
            "model": model_name,
            "clean_acc": clean_acc,
            "shuffled_acc": shuffled_acc,
            "delta_acc": delta_acc,
            "consistency": consistency,
        })

    # Print summary table
    print("\n" + "="*85)
    print(f"{'Model':<25} | {'Clean Acc (%)':<15} | {'Shuffled Acc (%)':<18} | {'Δ Acc (%)':<12} | {'Consistency (%)':<15}")
    print("="*85)
    for r in results:
        print(f"{r['model']:<25} | {r['clean_acc']:<15.2f} | {r['shuffled_acc']:<18.2f} | {r['delta_acc']:<12.2f} | {r['consistency']:<15.2f}")
    print("="*85)

if __name__ == "__main__":
    evaluate_patch_shuffling()