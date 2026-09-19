import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(nn.Module):
    """
    Standard MLP domain discriminator: 512 -> 256 -> 1
    Outputs a single logit (unnormalized probability of being target).
    """
    def __init__(self, in_features: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        # Pass through Gradient Reversal Layer first
        reversed_feat = ReverseLayerF.apply(x, alpha)
        return self.net(reversed_feat)