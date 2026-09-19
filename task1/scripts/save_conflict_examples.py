import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T
import open_clip

from common.seed import set_seed
from task1.models.backbones import ResNet50FeatureExtractor, ViTB16FeatureExtractor, CLIPFeatureExtractor
from task1.models.classifier_head import LinearProbe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_informative_examples():
    set_seed(6304)
    conflict_dir = Path("task1/data/cue_conflicts")
    with open(conflict_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    with open("task1/data/splits_seed6304.json", "r") as f:
        splits = json.load(f)
    classes = splits["classes"]

    # Load backbones and heads
    resnet = ResNet50FeatureExtractor().to(DEVICE)
    resnet_head = LinearProbe(2048).to(DEVICE)
    resnet_head.load_state_dict(torch.load("task1/models/ResNet-50_head.pth", map_location=DEVICE))
    resnet_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    vit = ViTB16FeatureExtractor().to(DEVICE)
    vit_head = LinearProbe(768).to(DEVICE)
    vit_head.load_state_dict(torch.load("task1/models/ViT-B_16_head.pth", map_location=DEVICE))
    vit_norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    clip_ext = CLIPFeatureExtractor().to(DEVICE)
    clip_model = clip_ext.model
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    prompts = [f"a photo of a {c}." for c in classes]
    text_tokens = tokenizer(prompts).to(DEVICE)
    text_feats = clip_model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.exp()
    clip_norm = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # Run inference across all conflict items to find specific behaviors
    eval_records = []
    for item in metadata:
        img_path = conflict_dir / item["file"]
        pil_img = Image.open(img_path).convert("RGB")
        tensor = T.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            r_pred = classes[torch.argmax(resnet_head(resnet(resnet_norm(tensor))), dim=-1).item()]
            v_pred = classes[torch.argmax(vit_head(vit(vit_norm(tensor))), dim=-1).item()]
            c_logits = logit_scale * (clip_ext(clip_norm(tensor)) @ text_feats.T)
            c_pred = classes[torch.argmax(c_logits, dim=-1).item()]

        eval_records.append({
            "item": item,
            "img": pil_img,
            "r_pred": r_pred,
            "v_pred": v_pred,
            "c_pred": c_pred,
            "shape": item["shape_class_name"],
            "texture": item["texture_class_name"]
        })

    # Search for targeted case categories
    selected = {}
    for rec in eval_records:
        s, t = rec["shape"], rec["texture"]
        r, v, c = rec["r_pred"], rec["v_pred"], rec["c_pred"]

        # Case 1: Texture Bias (ResNet predicts texture, ViT or CLIP predicts shape)
        if "texture_bias" not in selected and r == t and (v == s or c == s):
            selected["texture_bias"] = rec

        # Case 2: Unanimous Shape Agreement
        if "shape_agreement" not in selected and r == s and v == s and c == s:
            selected["shape_agreement"] = rec

        # Case 3: ViT/CLIP Disagreement
        if "model_disagreement" not in selected and v != c and (v == s or c == s):
            selected["model_disagreement"] = rec

        # Case 4: Other / Semantic Failure (Model predicts a class that is neither shape nor style)
        if "other_failure" not in selected and (r not in [s, t] or v not in [s, t] or c not in [s, t]):
            selected["other_failure"] = rec

    # Fallback if any category wasn't found
    candidates = list(selected.values())
    for rec in eval_records:
        if len(candidates) >= 4:
            break
        if rec not in candidates:
            candidates.append(rec)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    panel_titles = [
        "Texture Bias Case",
        "Shape Consensus",
        "Model Disagreement",
        "Low-Coverage Failure"
    ]

    for idx, rec in enumerate(candidates[:4]):
        ax = axes[idx]
        ax.imshow(rec["img"])
        title = (
            f"[{panel_titles[idx]}]\n"
            f"Shape: {rec['shape']} | Style: {rec['texture']}\n"
            f"ResNet-50: {rec['r_pred']}\n"
            f"ViT-B/16: {rec['v_pred']}\n"
            f"CLIP (ZS): {rec['c_pred']}"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    Path("report/figures").mkdir(parents=True, exist_ok=True)
    out_file = "report/figures/cue_conflict_examples.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Updated informative cue-conflict examples saved to {out_file}")

if __name__ == "__main__":
    save_informative_examples()