import torch
import torch.nn as nn
#a single linear layer for final classification
class LinearProbe(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)