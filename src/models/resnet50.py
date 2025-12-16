import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes: int = 2, pretrained: bool = True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
