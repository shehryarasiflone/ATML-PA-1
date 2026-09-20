import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for 32x32 inputs:
    - 3x3 conv1 with stride 1, padding 1
    - Identity maxpool (preserves spatial resolution)
    - Fully fine-tuned / randomly initialized
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = models.resnet18(weights=None)  # Random initialization

        # Modify first layer for 32x32 inputs
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = nn.Identity()

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor):
        feat = self.forward_features(x)
        logits = self.fc(feat)
        return feat, logits