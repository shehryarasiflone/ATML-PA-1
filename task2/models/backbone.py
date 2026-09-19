import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def freeze_bn_stats(model: nn.Module) -> None:
    """
    Enforces the assignment requirement:
    Freezes BatchNorm running mean and variance at pretrained ImageNet values
    while keeping learnable affine parameters (gamma and beta) trainable.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

class PACSResNet18(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        base = models.resnet18(weights=weights)

        # Retain convolutional layers and average pooling
        self.conv_layers = nn.Sequential(*list(base.children())[:-1])
        # Replace the 1000-class ImageNet head with a 7-class linear classifier
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        # x: [B, 3, 224, 224]
        feat = self.conv_layers(x)
        feat = torch.flatten(feat, 1)  # 512-dimensional feature vector
        logits = self.fc(feat)
        return feat, logits

    def train(self, mode: bool = True):
        """Override train to automatically freeze BatchNorm statistics."""
        super().train(mode)
        if mode:
            freeze_bn_stats(self)
        return self