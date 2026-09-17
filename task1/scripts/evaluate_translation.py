import json
import os
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset
from PIL import Image
import matplotlib.pyplot as plt
import open_clip

from common.seed import set_seed
from task1.models.backbones import ResNet50FeatureExtractor, ViTB16FeatureExtractor, CLIPFeatureExtractor
from task1.models.classifier_head import LinearProbe
from task1.data.transforms import apply_translation

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
    # Process in mini-batches
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

def evaluate_translation():
    set_seed(6304)
    with open("task1/data/splits_seed6304.json", "r") as f:
        splits = json.load(f)

    classes = splits["classes"]
    eval_indices = splits["eval_subset_indices"]
    raw_test = STL10(root="./data", split="test", download=False)

    # Cache clean evaluation subset images
    clean_images = [raw_test[idx][0].resize((224, 224)) for idx in eval_indices]
    targets = np.array([raw_test[idx][1] for idx in eval_indices])

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

    deltas = [0, 8, 16, 32]
    directions = ["left", "right", "up", "down"]
    summary_results = {m: {"acc": [], "consistency": []} for m in models_config}

    for model_name, cfg in models_config.items():
        print(f"\nEvaluating {model_name} on translation displacements...")
        head = None
        if not cfg["is_zs"]:
            head = LinearProbe(cfg["dim"]).to(DEVICE)
            head.load_state_dict(torch.load(cfg["head_path"], map_location=DEVICE))

        # 1. Clean reference predictions (delta = 0)
        clean_preds = get_predictions(cfg["extractor"], head, clean_images, cfg, is_clip_zs=cfg["is_zs"], class_names=classes)
        clean_acc = np.mean(clean_preds == targets) * 100.0

        for delta in deltas:
            if delta == 0:
                summary_results[model_name]["acc"].append(clean_acc)
                summary_results[model_name]["consistency"].append(100.0)
                continue

            dir_accs = []
            dir_consistencies = []

            for d in directions:
                shifted_imgs = [apply_translation(img, delta, d) for img in clean_images]
                preds = get_predictions(cfg["extractor"], head, shifted_imgs, cfg, is_clip_zs=cfg["is_zs"], class_names=classes)
                dir_accs.append(np.mean(preds == targets) * 100.0)
                dir_consistencies.append(np.mean(preds == clean_preds) * 100.0)

            # Average across the 4 cardinal directions
            avg_acc = float(np.mean(dir_accs))
            avg_con = float(np.mean(dir_consistencies))
            summary_results[model_name]["acc"].append(avg_acc)
            summary_results[model_name]["consistency"].append(avg_con)

    # Print summary table
    print("\n" + "="*85)
    print(f"{'Model':<25} | {'Delta':<6} | {'Avg Acc (%)':<15} | {'Avg Consistency (%)':<20}")
    print("="*85)
    for model_name in models_config:
        for idx, delta in enumerate(deltas):
            acc_val = summary_results[model_name]["acc"][idx]
            con_val = summary_results[model_name]["consistency"][idx]
            print(f"{model_name:<25} | {delta:<6} | {acc_val:<15.2f} | {con_val:<20.2f}")
        print("-" * 85)

    # Generate and save translation plot
    os.makedirs("report/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    markers = ['o-', 's--', '^-.', 'd:']
    for idx, (model_name, data) in enumerate(summary_results.items()):
        ax1.plot(deltas, data["acc"], markers[idx], label=model_name, linewidth=2)
        ax2.plot(deltas, data["consistency"], markers[idx], label=model_name, linewidth=2)

    ax1.set_title("Accuracy vs. Spatial Displacement")
    ax1.set_xlabel("Displacement $\delta$ (pixels)")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_xticks(deltas)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set_title("Prediction Consistency vs. Spatial Displacement")
    ax2.set_xlabel("Displacement $\delta$ (pixels)")
    ax2.set_ylabel("Consistency with Clean (%)")
    ax2.set_xticks(deltas)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = "report/figures/translation_curve.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot successfully saved to {plot_path}")

if __name__ == "__main__":
    evaluate_translation()