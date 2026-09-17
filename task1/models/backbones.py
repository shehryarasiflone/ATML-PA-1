import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
import open_clip

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.transforms = weights.transforms()
        base = models.resnet50(weights=weights)
        #we only need everything upto the final averagepooled layer
        self.features = nn.Sequential(*list(base.children())[:-1])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #flatten[Batch size, 2048, 1, 1]->[B, 2048]
        feat = self.features(x)
        return torch.flatten(feat, 1)

class ViTB16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.transforms = weights.transforms()
        base = models.vit_b_16(weights=weights)
        self.base = base
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through patch embeddings and transformer encoder to extract [CLS] token which is at index 0
        x = self.base._process_input(x)
        n = x.shape[0]
        batch_class_token = self.base.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.base.encoder(x)
        cls_token = x[:, 0]
        return self.base.heads.pre_logits(cls_token)

class CLIPFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model = model
        self.transforms = preprocess
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode and normalise for the final embeddings
        feats = self.model.encode_image(x)
        return feats / feats.norm(dim=-1, keepdim=True)