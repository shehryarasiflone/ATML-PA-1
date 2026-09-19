import torch
import torch.nn as nn
from task2.methods.dann import ReverseLayerF

class ConditionalDomainDiscriminator(nn.Module):
    """
    Discriminator taking multilinear conditioned representation h = vec(f_rev ⊗ p_detached)
    Input dimension: 512 * 7 = 3584
    """
    def __init__(self, in_features: int = 3584, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat: torch.Tensor, softmax_probs: torch.Tensor, alpha: float) -> torch.Tensor:
        b = feat.size(0)
        
        # 1. Reverse gradient ONLY through the feature representation
        feat_rev = ReverseLayerF.apply(feat, alpha)
        
        # 2. STRICTLY DETACH the class probabilities to prevent adversarial gradient
        # from corrupting the classifier head weights
        probs_detached = softmax_probs.detach()

        # 3. Explicit multilinear conditioning: [B, 7, 1] x [B, 1, 512] -> [B, 7, 512] -> [B, 3584]
        h = torch.bmm(probs_detached.unsqueeze(2), feat_rev.unsqueeze(1)).view(b, -1)
        return self.net(h)

def calc_entropy_weights(softmax_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes CDAN+E sample weights: w(x) = 1 + exp(-H(p))
    Weights must be completely detached from autograd graph.
    """
    with torch.no_grad():
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + eps), dim=1)
        weights = 1.0 + torch.exp(-entropy)
        weights = weights / weights.mean()
    return weights.detach()