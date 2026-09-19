import torch
import torch.nn as nn
from task2.methods.dann import ReverseLayerF

class ConditionalDomainDiscriminator(nn.Module):
    """
    Discriminator taking multilinear conditioned representation h = vec(f ⊗ p)
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
        # Batch size
        b = feat.size(0)
        
        # Explicit multilinear conditioning: h = vec(f ⊗ p)
        # [B, 512, 1] x [B, 1, 7] -> [B, 512, 7] -> [B, 3584]
        h = torch.bmm(feat.unsqueeze(2), softmax_probs.unsqueeze(1)).view(b, -1)

        # Gradient Reversal Layer
        h_rev = ReverseLayerF.apply(h, alpha)
        return self.net(h_rev)

def calc_entropy_weights(softmax_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes CDAN+E sample weights: w(x) = 1 + exp(-H(p))
    Downweights uncertain / ambiguous samples during adversarial alignment.
    """
    entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + eps), dim=1)
    weights = 1.0 + torch.exp(-entropy)
    # Normalize weights across the joint mini-batch to maintain loss scale
    weights = weights / weights.mean().detach()
    return weights