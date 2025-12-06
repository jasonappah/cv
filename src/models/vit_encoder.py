"""Vision Transformer encoder for frame-level feature extraction."""

import torch
import torch.nn as nn
import timm


class FrameViTEncoder(nn.Module):
    """Vision Transformer encoder for processing video frames.
    
    Takes a batch of frames and encodes each frame independently using a ViT.
    """
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        """Initialize ViT encoder.
        
        Args:
            model_name: Name of timm ViT model (default: vit_base_patch16_224)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load ViT from timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        # Get embedding dimension
        self.embed_dim = self.vit.embed_dim
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
               where B=batch, T=time (frames), C=channels, H=height, W=width
               
        Returns:
            embeddings: Tensor of shape (B, T, D)
                       where D is the embedding dimension (768 for vit_base)
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames at once: (B*T, C, H, W)
        x_flat = x.view(B * T, C, H, W)
        
        # Encode frames through ViT
        # ViT outputs (B*T, D) where D is embedding dimension
        embeddings_flat = self.vit(x_flat)
        
        # Reshape back to (B, T, D)
        embeddings = embeddings_flat.view(B, T, self.embed_dim)
        
        return embeddings

