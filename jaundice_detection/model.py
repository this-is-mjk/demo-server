"""
EfficientNet Model for Jaundice Detection
Supports both face and eye image classification.
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from torchvision.models import (
        efficientnet_b0, efficientnet_b1, efficientnet_b2,
        EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights
    )
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class JaundiceClassifier(nn.Module):
    """
    EfficientNet-based classifier for jaundice detection.
    Can be used for both face and eye images.
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create backbone
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # Remove classifier
            )
            num_features = self.backbone.num_features
            
        elif TORCHVISION_AVAILABLE:
            if 'b0' in model_name:
                weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
                self.backbone = efficientnet_b0(weights=weights)
                num_features = self.backbone.classifier[1].in_features
            elif 'b1' in model_name:
                weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
                self.backbone = efficientnet_b1(weights=weights)
                num_features = self.backbone.classifier[1].in_features
            elif 'b2' in model_name:
                weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
                self.backbone = efficientnet_b2(weights=weights)
                num_features = self.backbone.classifier[1].in_features
            else:
                weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
                self.backbone = efficientnet_b0(weights=weights)
                num_features = self.backbone.classifier[1].in_features
            
            self.backbone.classifier = nn.Identity()
        else:
            raise ImportError("Neither timm nor torchvision available")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(
    model_type: str = 'face',
    model_name: str = 'efficientnet_b0',
    pretrained: bool = True
) -> JaundiceClassifier:
    """
    Factory function to create a jaundice classifier.
    
    Args:
        model_type: 'face' or 'eyes'
        model_name: EfficientNet variant
        pretrained: Use pretrained weights
        
    Returns:
        JaundiceClassifier instance
    """
    dropout_rate = 0.3 if model_type == 'face' else 0.4
    
    return JaundiceClassifier(
        model_name=model_name,
        num_classes=2,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
