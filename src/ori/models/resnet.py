import torch
import torch.nn as nn
import torchvision
from .coral_layer import CoralLayer



class resnet(nn.Module):
    """
    ResNet model for ordinal regression that can process multiple images using attention fusion.
    
    For single images, this works like a standard CORAL/CORN ordinal regression model.
    For multiple images, it processes each image separately and uses an attention mechanism
    to fuse the features before making a final prediction. Attention is learned per instance.
    
    Supports four classification types:
    - coral: Ordinal classification using CORAL (Consistent Rank Logits)
    - corn: Ordinal classification using CORN (Consistent Ordinal Regression Neural Networks)
    - multiclass: Standard multiclass classification
    - ordinal_regression: Continuous ordinal regression
    """
    def __init__(self, num_classes, pretrained=True, classification_type="coral"):
        """
        Args:
            num_classes: Number of classes
            pretrained: Whether to use pretrained ResNet weights
            num_images: Number of images to process (1 or 2)
            field_names: Names of the fields used (for visualization)
            classification_type: Type of classification ("coral", "corn", "multiclass", "ordinal_regression")
        """
        super(resnet, self).__init__()
        # Load pretrained ResNet50 without the classification head
        base_model = torchvision.models.resnet50(weights='DEFAULT' if pretrained else None)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # The feature dimension from ResNet50
        self.feature_dim = 2048
        self.classification_type = classification_type
        self.num_classes = num_classes
        
        
        # Flexible classification head based on classification type
        if classification_type == "coral":
            # CORAL layer for ordinal classification
            self.classifier = CoralLayer(size_in=self.feature_dim, num_classes=num_classes, preinit_bias=True)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1) 
            )
        
        # Store attention weights for visualization
        self.attention_weights = None
        self.raw_attention_scores = None
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor or list of tensors.
               If num_images=1, x should be a tensor of shape [batch_size, channels, height, width]
               If num_images>1, x should be a list of tensors, each of shape [batch_size, channels, height, width]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Output logits/predictions
            attention_weights: (Optional) The attention weights used for fusion
        """

        features = self.features(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits