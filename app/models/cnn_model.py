from torch import nn
from torchvision import models


class RPSCNN(nn.Module):
    """Transfer-learning MobileNetV3 model adapted for Rock-Paper-Scissors classification."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
