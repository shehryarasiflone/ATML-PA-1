import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as T
import open_clip

from common.seed import set_seed
from common.metrics import compute_metrics
from task1.models.backbones import ResNet50FeatureExtractor, ViTB16FeatureExtractor, CLIPFeatureExtractor
from task1.models.classifier_head import LinearProbe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_base_transform(mean, std):
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

@torch.no_grad()
def extract_dataset_features(extractor, dataloader):
    extractor.eval()
    all_feats, all_labels = [], []
    for imgs, labels in dataloader:
        imgs = imgs.to(DEVICE)
        feats = extractor(imgs)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

def train_linear_head(train_feats, train_labels, val_feats, val_labels, in_features, num_classes=10):
    set_seed(6304)
    model = LinearProbe(in_features, num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    best_val_acc = -1.0
    patience_counter = 0
    best_weights = None

    for epoch in range(50):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_logits = model(val_feats.to(DEVICE))
            val_metrics = compute_metrics(val_logits, val_labels.to(DEVICE))
            val_acc = val_metrics["top1_acc"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    model.load_state_dict(best_weights)
    return model

@torch.no_grad()
def eval_clip_zero_shot(clip_extractor, test_loader, class_names):
    clip_model = clip_extractor.model
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # Generate text prompt embeddings: "a photo of a {class}."
    prompts = [f"a photo of a {c}." for c in class_names]
    text_tokens = tokenizer(prompts).to(DEVICE)
    text_feats = clip_model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    all_logits, all_targets = [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        img_feats = clip_extractor(imgs) # already unit normalized
        # Cosine similarity scaled by CLIP temperature
        logit_scale = clip_model.logit_scale.exp()
        logits = logit_scale * (img_feats @ text_feats.T)
        all_logits.append(logits.cpu())
        all_targets.append(labels)

    return compute_metrics(torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0))

def run_clean_baseline():
    set_seed(6304)
    with open("task1/data/splits_seed6304.json", "r") as f:
        splits = json.load(f)

    classes = splits["classes"]
    
    # STL-10 raw datasets
    raw_train = STL10(root="./data", split="train", download=False)
    raw_test = STL10(root="./data", split="test", download=False)

    models_config = {
        "ResNet-50": {
            "extractor": ResNet50FeatureExtractor().to(DEVICE),
            "dim": 2048,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "ViT-B/16": {
            "extractor": ViTB16FeatureExtractor().to(DEVICE),
            "dim": 768,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "CLIP (ViT-B/32)": {
            "extractor": CLIPFeatureExtractor().to(DEVICE),
            "dim": 512,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
    }

    results = {}

    for name, cfg in models_config.items():
        print(f"\n--- Processing {name} ---")
        transform = get_base_transform(cfg["mean"], cfg["std"])
        
        # Attach transform to dataset views
        raw_train.transform = transform
        raw_test.transform = transform

        train_sub = Subset(raw_train, splits["train_indices"])
        val_sub = Subset(raw_train, splits["val_indices"])
        eval_sub = Subset(raw_test, splits["eval_subset_indices"])

        train_loader = DataLoader(train_sub, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_sub, batch_size=64, shuffle=False)
        eval_loader = DataLoader(eval_sub, batch_size=64, shuffle=False)

        # 1. Feature extraction
        train_feats, train_labels = extract_dataset_features(cfg["extractor"], train_loader)
        val_feats, val_labels = extract_dataset_features(cfg["extractor"], val_loader)
        eval_feats, eval_labels = extract_dataset_features(cfg["extractor"], eval_loader)

        # 2. Linear Head Training
        head = train_linear_head(train_feats, train_labels, val_feats, val_labels, cfg["dim"])
        
        # Save head weights for subsequent intervention tasks
        torch.save(head.state_dict(), f"task1/models/{name.replace('/', '_')}_head.pth")

        # 3. Clean Evaluation
        head.eval()
        with torch.no_grad():
            clean_logits = head(eval_feats.to(DEVICE))
            metrics = compute_metrics(clean_logits, eval_labels.to(DEVICE))
            results[f"{name} (Linear Head)"] = metrics

        # 4. Zero-shot CLIP check
        if "CLIP" in name:
            clip_zs_metrics = eval_clip_zero_shot(cfg["extractor"], eval_loader, classes)
            results["CLIP (Zero-Shot)"] = clip_zs_metrics

    # Print Report Table
    print("\n" + "="*70)
    print(f"{'Model':<25} | {'Top-1 Acc (%)':<15} | {'Macro-F1 (%)':<15} | {'Mean Conf (%)':<15}")
    print("="*70)
    for model_name, m in results.items():
        print(f"{model_name:<25} | {m['top1_acc']:<15.2f} | {m['macro_f1']:<15.2f} | {m['mean_confidence']:<15.2f}")
    print("="*70)

if __name__ == "__main__":
    run_clean_baseline()