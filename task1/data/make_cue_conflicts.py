import os
import json
import urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.datasets import STL10
from PIL import Image

from common.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ AdaIN Architecture ------------------
class AdaINNet(nn.Module):
    def __init__(self):
        super().__init__()
        # VGG-19 encoder up to relu4_1
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Pretrained AdaIN Decoder network
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adain(self, content_feat, style_feat):
        size = content_feat.size()
        c_mean, c_std = self.calc_mean_std(content_feat)
        s_mean, s_std = self.calc_mean_std(style_feat)
        normalized = (content_feat - c_mean.expand(size)) / c_std.expand(size)
        return normalized * s_std.expand(size) + s_mean.expand(size)

    def forward(self, content, style, alpha=0.85):
        c_feat = self.encoder(content)
        s_feat = self.encoder(style)
        target = self.adain(c_feat, s_feat)
        target = alpha * target + (1 - alpha) * c_feat
        return self.decoder(target)

def download_adain_decoder(target_path="task1/data/adain_models/decoder.pth"):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path):
        print("Downloading pretrained AdaIN decoder weights...")
        url = "https://github.com/naoto0804/pytorch-AdaIN/raw/master/models/decoder.pth"
        urllib.request.urlretrieve(url, target_path)
        print("AdaIN decoder downloaded.")

# ------------------ Visual Rejection Rule ------------------
def evaluate_rejection_rule(content_tensor, stylized_tensor):
    """
    Evaluates visual validity before inference.
    Returns: (is_accepted: bool, reason: str)
    """
    # 1. Check pixel variance (contrast/mode collapse)
    var = stylized_tensor.var().item()
    if var < 0.005:
        return False, "low_contrast"

    # 2. Compute edge preservation via Sobel filter
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(DEVICE)

    # Convert to 1-channel luminance
    c_gray = (0.299 * content_tensor[:, 0:1] + 0.587 * content_tensor[:, 1:2] + 0.114 * content_tensor[:, 2:3])
    s_gray = (0.299 * stylized_tensor[:, 0:1] + 0.587 * stylized_tensor[:, 1:2] + 0.114 * stylized_tensor[:, 2:3])

    c_edge = torch.sqrt(nn.functional.conv2d(c_gray, sobel_x, padding=1)**2 + nn.functional.conv2d(c_gray, sobel_y, padding=1)**2)
    s_edge = torch.sqrt(nn.functional.conv2d(s_gray, sobel_x, padding=1)**2 + nn.functional.conv2d(s_gray, sobel_y, padding=1)**2)

    c_flat = c_edge.view(-1)
    s_flat = s_edge.view(-1)

    # Pearson correlation
    c_sub = c_flat - c_flat.mean()
    s_sub = s_flat - s_flat.mean()
    denom = (torch.sqrt((c_sub**2).sum()) * torch.sqrt((s_sub**2).sum())) + 1e-8
    r_edge = ((c_sub * s_sub).sum() / denom).item()

    if r_edge < 0.15:
        return False, "structural_collapse"
    if r_edge > 0.85:
        return False, "identity_failure"

    return True, "valid"

# ------------------ Main Generator ------------------
def generate_cue_conflicts():
    set_seed(6304)
    decoder_path = "task1/data/adain_models/decoder.pth"
    download_adain_decoder(decoder_path)

    model = AdaINNet().to(DEVICE)
    model.decoder.load_state_dict(torch.load(decoder_path, map_location=DEVICE))
    model.eval()

    output_dir = Path("task1/data/cue_conflicts")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("task1/data/splits_seed6304.json", "r") as f:
        splits = json.load(f)

    classes = splits["classes"]
    eval_indices = splits["eval_subset_indices"]
    raw_test = STL10(root="./data", split="test", download=False)

    # Organize images by class from the 500 evaluation subset
    class_to_imgs = {c: [] for c in range(10)}
    for idx in eval_indices:
        img, lbl = raw_test[idx]
        img_resized = img.resize((224, 224))
        class_to_imgs[lbl].append(img_resized)

    # Five unordered semantic pairs that do have some sort of sembalance
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

    print("Generating 200 cue-conflict images across 5 bidirectional pairs...")

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

                with torch.no_grad():
                    stylized_t = model(c_t, s_t, alpha=0.85)
                    stylized_t = torch.clamp(stylized_t, 0.0, 1.0)

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

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- Visual Rejection Rule Summary ---")
    print(f"Accepted Valid Conflicts : {rejection_counts['valid']} / {len(metadata)}")
    print(f"Structural Collapses     : {rejection_counts['structural_collapse']}")
    print(f"Identity Failures        : {rejection_counts['identity_failure']}")
    print(f"Low Contrast Collapses   : {rejection_counts['low_contrast']}")
    print(f"Conflicts written to {output_dir}")

if __name__ == "__main__":
    generate_cue_conflicts()