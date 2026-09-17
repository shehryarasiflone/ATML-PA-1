import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.datasets import STL10
from PIL import Image

from common.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ VGG Style Transfer Feature Extractor ------------------
class VGGStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Official PyTorch VGG19 weights (guaranteed availability)
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        # Layers: conv1_1 (0), conv2_1 (5), conv3_1 (10), conv4_1 (19), conv5_1 (28)
        self.style_layers = [0, 5, 10, 19, 28]
        self.content_layer = 19  # relu4_1 / conv4_1

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

    def forward(self, x):
        # Normalize input to ImageNet distribution
        h = (x - self.mean) / self.std
        style_feats = []
        content_feat = None

        for idx, layer in enumerate(self.vgg):
            h = layer(h)
            if idx in self.style_layers:
                style_feats.append(h)
            if idx == self.content_layer:
                content_feat = h

        return content_feat, style_feats

def calc_gram_matrix(feat):
    b, c, h, w = feat.size()
    f = feat.view(b * c, h * w)
    gram = torch.mm(f, f.t())
    return gram.div(c * h * w)

def transfer_style(vgg_net, content_img, style_img, steps=35, content_weight=1.0, style_weight=5e4):
    """
    Fast iterative style transfer producing high-quality cue conflict stimuli.
    """
    with torch.no_grad():
        target_content, _ = vgg_net(content_img)
        _, style_feats = vgg_net(style_img)
        target_grams = [calc_gram_matrix(s) for s in style_feats]

    # Initialize synthesized image from content image to preserve silhouette
    target = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=0.05)

    for _ in range(steps):
        optimizer.zero_grad()
        c_feat, s_feats = vgg_net(target)

        # Content loss (preserves global shape)
        c_loss = nn.functional.mse_loss(c_feat, target_content)

        # Style loss (imposes texture and color distribution)
        s_loss = 0.0
        for s_f, t_gram in zip(s_feats, target_grams):
            curr_gram = calc_gram_matrix(s_f)
            s_loss += nn.functional.mse_loss(curr_gram, t_gram)

        loss = content_weight * c_loss + style_weight * s_loss
        loss.backward()
        optimizer.step()

        # Keep pixels within valid RGB range [0, 1]
        with torch.no_grad():
            target.clamp_(0.0, 1.0)

    return target.detach()

# ------------------ Visual Rejection Rule ------------------
def evaluate_rejection_rule(content_tensor, stylized_tensor):
    """
    Evaluates visual validity before model evaluation.
    Returns: (is_accepted: bool, reason: str)
    """
    # 1. Pixel variance (low-contrast check)
    var = stylized_tensor.var().item()
    if var < 0.005:
        return False, "low_contrast"

    # 2. Edge correlation via Sobel filter
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=DEVICE).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=DEVICE).view(1, 1, 3, 3)

    c_gray = 0.299 * content_tensor[:, 0:1] + 0.587 * content_tensor[:, 1:2] + 0.114 * content_tensor[:, 2:3]
    s_gray = 0.299 * stylized_tensor[:, 0:1] + 0.587 * stylized_tensor[:, 1:2] + 0.114 * stylized_tensor[:, 2:3]

    c_edge = torch.sqrt(nn.functional.conv2d(c_gray, sobel_x, padding=1)**2 + nn.functional.conv2d(c_gray, sobel_y, padding=1)**2)
    s_edge = torch.sqrt(nn.functional.conv2d(s_gray, sobel_x, padding=1)**2 + nn.functional.conv2d(s_gray, sobel_y, padding=1)**2)

    c_flat = c_edge.view(-1)
    s_flat = s_edge.view(-1)

    c_sub = c_flat - c_flat.mean()
    s_sub = s_flat - s_flat.mean()
    denom = (torch.sqrt((c_sub**2).sum()) * torch.sqrt((s_sub**2).sum())) + 1e-8
    r_edge = ((c_sub * s_sub).sum() / denom).item()

    if r_edge < 0.15:
        return False, "structural_collapse"
    if r_edge > 0.85:
        return False, "identity_failure"

    return True, "valid"

# ------------------ Generator Execution ------------------
def generate_cue_conflicts():
    set_seed(6304)
    vgg_net = VGGStyleNet().to(DEVICE)

    output_dir = Path("task1/data/cue_conflicts")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("task1/data/splits_seed6304.json", "r") as f:
        splits = json.load(f)

    classes = splits["classes"]
    eval_indices = splits["eval_subset_indices"]
    raw_test = STL10(root="./data", split="test", download=False)

    # Cache evaluation images grouped by class
    class_to_imgs = {c: [] for c in range(10)}
    for idx in eval_indices:
        img, lbl = raw_test[idx]
        img_resized = img.resize((224, 224))
        class_to_imgs[lbl].append(img_resized)

    # Five verified semantic pairs in STL-10
    class_pairs = [
        ("airplane", "bird"),
        ("cat", "dog"),
        ("deer", "horse"),
        ("car", "truck"),
        ("car", "ship"),
    ]

    target_per_dir = 20
    metadata = []
    rejection_counts = {"valid": 0, "structural_collapse": 0, "identity_failure": 0, "low_contrast": 0}
    to_tensor = T.ToTensor()

    print("Generating 200 cue-conflict images via VGG style transfer...")

    for c1_name, c2_name in class_pairs:
        c1_idx = classes.index(c1_name)
        c2_idx = classes.index(c2_name)

        directions = [
            (c1_idx, c1_name, c2_idx, c2_name),
            (c2_idx, c2_name, c1_idx, c1_name),
        ]

        for content_cls, c_name, style_cls, s_name in directions:
            accepted_count = 0
            c_imgs = class_to_imgs[content_cls]
            s_imgs = class_to_imgs[style_cls]

            i, j = 0, 0
            while accepted_count < target_per_dir and i < len(c_imgs):
                c_img = c_imgs[i]
                s_img = s_imgs[j % len(s_imgs)]

                c_t = to_tensor(c_img).unsqueeze(0).to(DEVICE)
                s_t = to_tensor(s_img).unsqueeze(0).to(DEVICE)

                stylized_t = transfer_style(vgg_net, c_t, s_t)

                valid, reason = evaluate_rejection_rule(c_t, stylized_t)
                rejection_counts[reason] += 1

                if valid:
                    filename = f"conflict_{c_name}_style_{s_name}_{accepted_count:02d}.png"
                    save_img = TF.to_pil_image(stylized_t.squeeze(0).cpu())
                    save_img.save(output_dir / filename)

                    metadata.append({
                        "file": filename,
                        "shape_class_id": content_cls,
                        "shape_class_name": c_name,
                        "texture_class_id": style_cls,
                        "texture_class_name": s_name,
                    })
                    accepted_count += 1

                i += 1
                j += 1

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- Visual Rejection Rule Summary ---")
    print(f"Accepted Valid Conflicts : {rejection_counts['valid']} / {len(metadata)}")
    print(f"Structural Collapses     : {rejection_counts['structural_collapse']}")
    print(f"Identity Failures        : {rejection_counts['identity_failure']}")
    print(f"Low Contrast Collapses   : {rejection_counts['low_contrast']}")
    print(f"All images saved to {output_dir}")

if __name__ == "__main__":
    generate_cue_conflicts()