import json
import torch
import numpy as np
from torchvision.datasets import STL10
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import open_clip

from common.seed import set_seed
from task1.models.backbones import ResNet50FeatureExtractor, ViTB16FeatureExtractor, CLIPFeatureExtractor
from task1.models.classifier_head import LinearProbe
from task1.data.transforms import apply_grayscale, apply_hue_rotation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_eval_pipeline(mean, std, transform_fn=None):
    def transform(img):
        img = img.resize((224, 224))
        if transform_fn is not None:
            img = transform_fn(img)
        tensor = T.ToTensor()(img)
        return T.Normalize(mean=mean, std=std)(tensor)
    return transform

@torch.no_grad()
def get_predictions(extractor, head, loader, is_clip_zs=False, class_names=None):
    extractor.eval()
    if head is not None:
        head.eval()
    
    all_preds = []
    all_targets = []

    if is_clip_zs:
        clip_model = extractor.model
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        prompts = [f"a photo of a {c}." for c in class_names]
        text_tokens = tokenizer(prompts).to(DEVICE)
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        logit_scale = clip_model.logit_scale.exp()

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        feats = extractor(imgs)
        if is_clip_zs:
            logits = logit_scale * (feats @ text_feats.T)
        else:
            logits = head(feats)
        preds = torch.argmax(logits, dim=-1).cpu()
        all_preds.extend(preds.numpy())
        all_targets.extend(labels.numpy())

    return np.array(all_preds), np.array(all_targets)

def evaluate_color_interventions():
    set_seed(6304)
    with open("task1/data/splits_seed6304.json", "r") as f:
        splits = json.load(f)

    classes = splits["classes"]
    eval_indices = splits["eval_subset_indices"]
    raw_test = STL10(root="./data", split="test", download=False)

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

    eval_modes = [
        ("Clean", None),
        ("Grayscale", apply_grayscale),
        ("Hue Rotation (180 deg)", apply_hue_rotation),
    ]

    results_table = []

    for name, cfg in models_config.items():
        extractor = cfg["extractor"]
        head = LinearProbe(cfg["dim"]).to(DEVICE)
        head.load_state_dict(torch.load(cfg["head_path"], map_location=DEVICE))

        # Evaluate linear head
        model_runs = [("Linear Head", head, False)]
        if "CLIP" in name:
            model_runs.append(("Zero-Shot", None, True))

        for tag, m_head, is_zs in model_runs:
            full_name = f"{name} ({tag})"
            clean_preds = None

            for mode_name, transform_fn in eval_modes:
                pipeline = get_eval_pipeline(cfg["mean"], cfg["std"], transform_fn)
                raw_test.transform = pipeline
                eval_sub = Subset(raw_test, eval_indices)
                loader = DataLoader(eval_sub, batch_size=64, shuffle=False)

                preds, targets = get_predictions(extractor, m_head, loader, is_clip_zs=is_zs, class_names=classes)
                acc = np.mean(preds == targets) * 100.0

                if mode_name == "Clean":
                    clean_preds = preds
                    clean_acc = acc
                    results_table.append({
                        "model": full_name,
                        "intervention": "Clean",
                        "acc": clean_acc,
                        "delta_acc": 0.0,
                        "consistency": 100.0,
                    })
                else:
                    consistency = np.mean(preds == clean_preds) * 100.0
                    delta_acc = acc - clean_acc
                    results_table.append({
                        "model": full_name,
                        "intervention": mode_name,
                        "acc": acc,
                        "delta_acc": delta_acc,
                        "consistency": consistency,
                    })

    # Display results table
    print("\n" + "="*85)
    print(f"{'Model':<26} | {'Intervention':<22} | {'Acc (%)':<10} | {'Δ Acc (%)':<10} | {'Consistency (%)':<15}")
    print("="*85)
    for r in results_table:
        print(f"{r['model']:<26} | {r['intervention']:<22} | {r['acc']:<10.2f} | {r['delta_acc']:<10.2f} | {r['consistency']:<15.2f}")
    print("="*85)

if __name__ == "__main__":
    evaluate_color_interventions()