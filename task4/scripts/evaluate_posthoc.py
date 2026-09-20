import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from common.seed import set_seed
from task4.data.cifar_loaders import CIFAR_EVAL_TRANSFORM, get_osr_dataloaders
from task4.models.resnet_cifar import ResNet18CIFAR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def extract_features_and_logits(model, loader):
    model.eval()
    feats, logits = [], []
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        f, z = model(imgs)
        feats.append(f.cpu())
        logits.append(z.cpu())
    return torch.cat(feats, dim=0), torch.cat(logits, dim=0)

def compute_mahalanobis_parameters(model, split_path="task4/splits/cifar_osr_seed6304.json"):
    """
    Computes class means mu_c and shared diagonal covariance Sigma from
    unaugmented CIFAR-10 training features with 1e-6 added to the diagonal.
    """
    with open(split_path, "r") as f:
        manifest = json.load(f)

    c10_root = manifest.get("c10_root", "./data")
    c10_train_raw = CIFAR10(root=c10_root, train=True, download=False, transform=CIFAR_EVAL_TRANSFORM)
    train_ds = Subset(c10_train_raw, manifest["train_indices"])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=2)

    model.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in train_loader:
            f, _ = model(imgs.to(DEVICE))
            feats_list.append(f.cpu())
            labels_list.append(labels)

    feats = torch.cat(feats_list, dim=0)    # [45000, 512]
    labels = torch.cat(labels_list, dim=0)  # [45000]

    # Compute per-class means mu_c
    means = torch.stack([feats[labels == c].mean(dim=0) for c in range(10)])  # [10, 512]

    # Compute shared diagonal covariance Sigma
    diff = feats - means[labels]  # [45000, 512]
    var = torch.mean(diff ** 2, dim=0) + 1e-6  # [512]

    return means.to(DEVICE), var.to(DEVICE)

@torch.no_grad()
def compute_scores(feats, logits, means, var):
    feats = feats.to(DEVICE)
    logits = logits.to(DEVICE)

    # 1. MSP: 1 - max softmax probability
    probs = F.softmax(logits, dim=-1)
    msp = 1.0 - torch.max(probs, dim=-1)[0]

    # 2. MLS: -max logit
    mls = -torch.max(logits, dim=-1)[0]

    # 3. Energy: -logsumexp
    energy = -torch.logsumexp(logits, dim=-1)

    # 4. Mahalanobis distance: min_c (f - mu_c)^T Sigma^-1 (f - mu_c)
    # feats: [N, 512], means: [10, 512], var: [512]
    # (x - mu)^2 / var
    diff = feats.unsqueeze(1) - means.unsqueeze(0)  # [N, 10, 512]
    dist_per_class = torch.sum((diff ** 2) / var.unsqueeze(0).unsqueeze(0), dim=-1)  # [N, 10]
    mahalanobis = torch.min(dist_per_class, dim=-1)[0]  # [N]

    return {
        "MSP": msp.cpu().numpy(),
        "MLS": mls.cpu().numpy(),
        "Energy": energy.cpu().numpy(),
        "Mahalanobis": mahalanobis.cpu().numpy()
    }

def evaluate_posthoc():
    set_seed(6304)
    _, val_loader, test_loader, near_loader, far_loader = get_osr_dataloaders()

    model = ResNet18CIFAR(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load("task4/checkpoints/vanilla_best.pth", map_location=DEVICE))

    print("Extracting features and logits across all splits...")
    val_f, val_z = extract_features_and_logits(model, val_loader)
    test_f, test_z = extract_features_and_logits(model, test_loader)
    near_f, near_z = extract_features_and_logits(model, near_loader)
    far_f, far_z = extract_features_and_logits(model, far_loader)

    print("Computing class means and covariance on unaugmented training data...")
    means, var = compute_mahalanobis_parameters(model)

    val_scores = compute_scores(val_f, val_z, means, var)
    test_scores = compute_scores(test_f, test_z, means, var)
    near_scores = compute_scores(near_f, near_z, means, var)
    far_scores = compute_scores(far_f, far_z, means, var)

    score_names = ["MSP", "MLS", "Energy", "Mahalanobis"]

    print("\n" + "="*110)
    print("TASK 4 — STEP 2: POST-HOC NOVELTY SCORES ON VANILLA MODEL")
    print("="*110)
    header = (f"{'Score':<13} | {'AUROC (Near)':<13} | {'AUROC (Far)':<13} | {'AUROC (All)':<13} | "
              f"{'Test Acc (TPR)':<15} | {'Near Rej Rate':<14} | {'Far Rej Rate':<13} | {'FPR@95TPR'}")
    print(header)
    print("-" * 110)

    for s in score_names:
        u_val = val_scores[s]
        u_test = test_scores[s]
        u_near = near_scores[s]
        u_far = far_scores[s]
        u_all = np.concatenate([u_near, u_far])

        # AUROC computation (y=0 for known test, y=1 for unknown)
        y_near = np.concatenate([np.zeros(len(u_test)), np.ones(len(u_near))])
        score_near = np.concatenate([u_test, u_near])
        auroc_near = roc_auc_score(y_near, score_near) * 100.0

        y_far = np.concatenate([np.zeros(len(u_test)), np.ones(len(u_far))])
        score_far = np.concatenate([u_test, u_far])
        auroc_far = roc_auc_score(y_far, score_far) * 100.0

        y_all = np.concatenate([np.zeros(len(u_test)), np.ones(len(u_all))])
        score_all = np.concatenate([u_test, u_all])
        auroc_all = roc_auc_score(y_all, score_all) * 100.0

        # Calibrate threshold at 95th percentile of validation unknownness
        # (95% of validation knowns have u <= tau, i.e., are accepted)
        tau = np.percentile(u_val, 95)

        # Test evaluation at threshold tau
        tpr = np.mean(u_test <= tau) * 100.0          # Known test acceptance rate
        near_rej = np.mean(u_near > tau) * 100.0      # Near rejection rate
        far_rej = np.mean(u_far > tau) * 100.0        # Far rejection rate
        fpr_all = np.mean(u_all <= tau) * 100.0       # FPR@95TPR (fraction of all unknowns accepted)

        print(f"{s:<13} | {auroc_near:>11.2f}% | {auroc_far:>11.2f}% | {auroc_all:>11.2f}% | "
              f"{tpr:>13.2f}% | {near_rej:>12.2f}% | {far_rej:>11.2f}% | {fpr_all:>8.2f}%")

    print("="*110)

if __name__ == "__main__":
    evaluate_posthoc()