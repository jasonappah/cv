"""Complete gesture recognition model combining ViT encoder and SSM backbone."""

import torch
import torch.nn as nn
from .vit_encoder import FrameViTEncoder
from .ssm_backbone import SimpleSSMBackbone


class GestureModel(nn.Module):
    """Complete gesture recognition model.
    
    Architecture: ViT Encoder → SSM Backbone → Temporal Pooling → Classifier
    """
    
    def __init__(self, num_classes=27, vit_model='vit_base_patch16_224', 
                 vit_pretrained=True, ssm_layers=4, ssm_d_state=16, 
                 ssm_d_conv=4, ssm_expand=2):
        """Initialize gesture recognition model.
        
        Args:
            num_classes: Number of gesture classes (27 for Jester)
            vit_model: Name of timm ViT model
            vit_pretrained: Whether to use pretrained ViT weights
            ssm_layers: Number of SSM blocks
            ssm_d_state: SSM state dimension
            ssm_d_conv: SSM convolution kernel size
            ssm_expand: SSM expansion factor
        """
        super().__init__()
        
        # ViT encoder for frame-level features
        self.encoder = FrameViTEncoder(
            model_name=vit_model,
            pretrained=vit_pretrained
        )
        
        # Get embedding dimension from encoder
        d_model = self.encoder.embed_dim
        
        # SSM backbone for temporal modeling
        self.ssm_backbone = SimpleSSMBackbone(
            d_model=d_model,
            num_layers=ssm_layers,
            d_state=ssm_d_state,
            d_conv=ssm_d_conv,
            expand=ssm_expand,
            pool_output=True  # Output pooled representation
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
               where B=batch, T=time (frames), C=channels, H=height, W=width
               
        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        # Encode frames: (B, T, C, H, W) → (B, T, D)
        frame_embeddings = self.encoder(x)
        
        # Temporal modeling: (B, T, D) → (B, D)
        temporal_features = self.ssm_backbone(frame_embeddings)
        
        # Classification: (B, D) → (B, num_classes)
        logits = self.classifier(temporal_features)
        
        return logits

