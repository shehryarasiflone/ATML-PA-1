import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for 32x32 inputs with stage splits for manifold mixup:
    - Stage 1: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2
    - Stage 2: layer3 -> layer4 -> avgpool -> flatten
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = models.resnet18(weights=None)

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

    def forward_features_stage1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_features_stage2(self, h: torch.Tensor) -> torch.Tensor:
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.avgpool(h)
        return torch.flatten(h, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features_stage1(x)
        return self.forward_features_stage2(h)

    def forward(self, x: torch.Tensor):
        feat = self.forward_features(x)
        logits = self.fc(feat)
        return feat, logits