import json
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import open_clip

from common.seed import set_seed
from task1.models.backbones import ResNet50FeatureExtractor, ViTB16FeatureExtractor, CLIPFeatureExtractor
from task1.models.classifier_head import LinearProbe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_cue_conflicts():
    set_seed(6304)
    conflict_dir = Path("task1/data/cue_conflicts")
    with open(conflict_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    with open("task1/data/splits_seed6304.json", "r") as f:
        splits = json.load(f)
    classes = splits["classes"]

    models_config = {
        "ResNet-50": {
            "extractor": ResNet50FeatureExtractor().to(DEVICE),
            "head_path": "task1/models/ResNet-50_head.pth",
            "dim": 2048,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "ViT-B/16": {
            "extractor": ViTB16FeatureExtractor().to(DEVICE),
            "head_path": "task1/models/ViT-B_16_head.pth",
            "dim": 768,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "CLIP (ViT-B/32)": {
            "extractor": CLIPFeatureExtractor().to(DEVICE),
            "head_path": "task1/models/CLIP (ViT-B_32)_head.pth",
            "dim": 512,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
    }

    results = []

    for name, cfg in models_config.items():
        extractor = cfg["extractor"]
        head = LinearProbe(cfg["dim"]).to(DEVICE)
        head.load_state_dict(torch.load(cfg["head_path"], map_location=DEVICE))

        runs = [("Linear Head", head, False)]
        if "CLIP" in name:
            runs.append(("Zero-Shot", None, True))

        norm_transform = T.Normalize(mean=cfg["mean"], std=cfg["std"])

        for tag, m_head, is_zs in runs:
            full_name = f"{name} ({tag})"
            n_shape = 0
            n_texture = 0
            n_other = 0

            if is_zs:
                clip_model = extractor.model
                tokenizer = open_clip.get_tokenizer('ViT-B-32')
                prompts = [f"a photo of a {c}." for c in classes]
                text_tokens = tokenizer(prompts).to(DEVICE)
                text_feats = clip_model.encode_text(text_tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                logit_scale = clip_model.logit_scale.exp()

            for item in metadata:
                img_path = conflict_dir / item["file"]
                img = Image.open(img_path).convert("RGB")
                tensor = T.ToTensor()(img)
                norm_img = norm_transform(tensor).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    feats = extractor(norm_img)
                    if is_zs:
                        logits = logit_scale * (feats @ text_feats.T)
                    else:
                        logits = m_head(feats)
                    pred = torch.argmax(logits, dim=-1).item()

                if pred == item["shape_class_id"]:
                    n_shape += 1
                elif pred == item["texture_class_id"]:
                    n_texture += 1
                else:
                    n_other += 1

            n_total = len(metadata)
            denom = n_shape + n_texture
            shape_bias = (n_shape / denom * 100.0) if denom > 0 else 0.0
            coverage = (denom / n_total * 100.0)

            results.append({
                "model": full_name,
                "n_shape": n_shape,
                "n_texture": n_texture,
                "n_other": n_other,
                "shape_bias": shape_bias,
                "coverage": coverage,
            })

    print("\n" + "="*85)
    print(f"{'Model':<26} | {'N_shape':<8} | {'N_texture':<10} | {'N_other':<8} | {'Shape Bias (%)':<15} | {'Coverage (%)':<12}")
    print("="*85)
    for r in results:
        print(f"{r['model']:<26} | {r['n_shape']:<8} | {r['n_texture']:<10} | {r['n_other']:<8} | {r['shape_bias']:<15.2f} | {r['coverage']:<12.2f}")
    print("="*85)

if __name__ == "__main__":
    evaluate_cue_conflicts()