import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Computes Top-1 Accuracy, Macro-F1, and Mean Maximum Confidence.
    
    Args:
        logits: Tensor of shape [N, num_classes]
        targets: Tensor of shape [N]
    """
    probs = torch.softmax(logits, dim=-1)
    confidences, preds = torch.max(probs, dim=-1)

    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    conf_np = confidences.detach().cpu().numpy()

    top1_acc = np.mean(preds_np == targets_np) * 100.0
    macro_f1 = f1_score(targets_np, preds_np, average="macro") * 100.0
    mean_conf = np.mean(conf_np) * 100.0

    return {
        "top1_acc": float(top1_acc),
        "macro_f1": float(macro_f1),
        "mean_confidence": float(mean_conf),
    }