import json
import os
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import STL10
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from common.seed import set_seed
from task1.models.backbones import ResNet50FeatureExtractor, ViTB16FeatureExtractor, CLIPFeatureExtractor
from task1.data.transforms import apply_grayscale, apply_translation, apply_patch_shuffle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def extract_features(extractor, pil_images, cfg):
    extractor.eval()
    norm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])
    feats = []
    batch_size = 64
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i + batch_size]
        tensors = torch.stack([norm(img) for img in batch]).to(DEVICE)
        out = extractor(tensors).cpu()
        feats.append(out)
    return torch.cat(feats, dim=0)

def compute_cosine_stability(feats_clean, feats_trans):
    # Normalized inner product along feature dimension
    c_norm = feats_clean / (feats_clean.norm(dim=-1, keepdim=True) + 1e-8)
    t_norm = feats_trans / (feats_trans.norm(dim=-1, keepdim=True) + 1e-8)
    cosine_sim = (c_norm * t_norm).sum(dim=-1)
    return float(cosine_sim.mean().item())

def evaluate_representations():
    set_seed(6304)
    with open("task1/data/splits_seed6304.json", "r") as f:
        splits = json.load(f)

    classes = splits["classes"]
    eval_indices = splits["eval_subset_indices"]
    raw_test = STL10(root="./data", split="test", download=False)

    # 1. Prepare clean images & labels (500 subset)
    clean_images = [raw_test[idx][0].resize((224, 224)) for idx in eval_indices]
    targets = np.array([raw_test[idx][1] for idx in eval_indices])

    # 2. Interventions on evaluation subset
    gray_images = [apply_grayscale(img) for img in clean_images]
    trans32_images = [apply_translation(img, delta=32, direction="right") for img in clean_images]
    shuffle_images = [apply_patch_shuffle(img, seed=6304 + i) for i, img in enumerate(clean_images)]

    # 3. Cue-conflict pairs (pairing conflict image with its original clean content image)
    conflict_dir = Path("task1/data/cue_conflicts")
    with open(conflict_dir / "metadata.json", "r") as f:
        conflict_meta = json.load(f)

    conflict_imgs = [Image.open(conflict_dir / item["file"]).convert("RGB") for item in conflict_meta]
    
    # Map each conflict to its clean content image from evaluation subset
    class_to_clean = {c: [] for c in range(10)}
    for idx in eval_indices:
        class_to_clean[raw_test[idx][1]].append(raw_test[idx][0].resize((224, 224)))

    content_clean_pairs = []
    class_counters = {c: 0 for c in range(10)}
    for item in conflict_meta:
        c_cls = item["shape_class_id"]
        c_idx = class_counters[c_cls] % len(class_to_clean[c_cls])
        content_clean_pairs.append(class_to_clean[c_cls][c_idx])
        class_counters[c_cls] += 1

    backbones = {
        "ResNet-50": {
            "extractor": ResNet50FeatureExtractor().to(DEVICE),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "ViT-B/16": {
            "extractor": ViTB16FeatureExtractor().to(DEVICE),
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "CLIP (ViT-B/32)": {
            "extractor": CLIPFeatureExtractor().to(DEVICE),
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        }
    }

    stability_table = []
    tsne_data = {}

    for name, cfg in backbones.items():
        print(f"\nComputing representations for {name}...")
        f_clean = extract_features(cfg["extractor"], clean_images, cfg)
        f_gray = extract_features(cfg["extractor"], gray_images, cfg)
        f_trans = extract_features(cfg["extractor"], trans32_images, cfg)
        f_shuf = extract_features(cfg["extractor"], shuffle_images, cfg)

        # Cue conflict features
        f_conf_clean = extract_features(cfg["extractor"], content_clean_pairs, cfg)
        f_conf = extract_features(cfg["extractor"], conflict_imgs, cfg)

        # Cosine Stability
        cos_gray = compute_cosine_stability(f_clean, f_gray)
        cos_trans = compute_cosine_stability(f_clean, f_trans)
        cos_shuf = compute_cosine_stability(f_clean, f_shuf)
        cos_conf = compute_cosine_stability(f_conf_clean, f_conf)

        stability_table.append({
            "model": name,
            "grayscale": cos_gray,
            "translation_32": cos_trans,
            "patch_shuffle": cos_shuf,
            "cue_conflict": cos_conf,
        })

        # Fit joint t-SNE for Clean vs. Patch Shuffling
        print(f"Fitting t-SNE joint projection for {name}...")
        combined_feats = torch.cat([f_clean, f_shuf], dim=0).numpy()
        tsne = TSNE(n_components=2, perplexity=30, random_state=6304, max_iter=1000)
        proj = tsne.fit_transform(combined_feats)
        tsne_data[name] = proj

    # Print Cosine Stability Table
    print("\n" + "="*85)
    print(f"{'Backbone':<18} | {'Grayscale':<12} | {'Trans (32px)':<14} | {'Patch Shuffle':<15} | {'Cue Conflict':<12}")
    print("="*85)
    for r in stability_table:
        print(f"{r['model']:<18} | {r['grayscale']:<12.4f} | {r['translation_32']:<14.4f} | {r['patch_shuffle']:<15.4f} | {r['cue_conflict']:<12.4f}")
    print("="*85)

    # Plot joint t-SNE figures
    os.makedirs("report/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    cmap = plt.get_cmap("tab10")

    for idx, (name, proj) in enumerate(tsne_data.items()):
        ax = axes[idx]
        n_pts = len(clean_images)
        clean_proj = proj[:n_pts]
        trans_proj = proj[n_pts:]

        for c in range(10):
            mask = (targets == c)
            ax.scatter(clean_proj[mask, 0], clean_proj[mask, 1], color=cmap(c), marker="o", alpha=0.7, s=25, label=f"{classes[c]}" if idx == 0 else "")
            ax.scatter(trans_proj[mask, 0], trans_proj[mask, 1], color=cmap(c), marker="x", alpha=0.5, s=30)

        ax.set_title(f"{name}: Clean (o) vs. Shuffled (x)", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=11)
    plt.tight_layout()
    plot_path = "report/figures/tsne_representations.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nt-SNE visualization saved to {plot_path}")

if __name__ == "__main__":
    evaluate_representations()