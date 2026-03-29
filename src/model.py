# Responsibilities:
# - Define model architecture
# - Load pretrained weights
# - Modify final layer for N species

import torchvision.models as models
import torch.nn as nn

class Model:
    def __init__(self): ...

    @ staticmethod
    def build_model(num_classes: int, freeze_backbone: bool=False):
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False  # freeze backbone

        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # lower dropout when only training classifier
            nn.Linear(num_features, num_classes)
        )
        return model