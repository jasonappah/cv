import torch
import torch.nn as nn
import timm

class FrameViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        embeddings_flat = self.vit(x_flat)
        embeddings = embeddings_flat.view(B, T, self.embed_dim)
        return embeddings