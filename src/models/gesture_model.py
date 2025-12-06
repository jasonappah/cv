import torch
import torch.nn as nn
from .vit_encoder import FrameViTEncoder
from .ssm_backbone import SimpleSSMBackbone

class GestureModel(nn.Module):
    def __init__(self, num_classes=27, vit_model='vit_base_patch16_224', 
                 vit_pretrained=True, ssm_layers=4, ssm_d_state=16, 
                 ssm_d_conv=4, ssm_expand=2):
        super().__init__()
        self.encoder = FrameViTEncoder(model_name=vit_model, pretrained=vit_pretrained)
        d_model = self.encoder.embed_dim
        self.ssm_backbone = SimpleSSMBackbone(
            d_model=d_model, num_layers=ssm_layers, d_state=ssm_d_state,
            d_conv=ssm_d_conv, expand=ssm_expand, pool_output=True
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        frame_embeddings = self.encoder(x)
        temporal_features = self.ssm_backbone(frame_embeddings)
        logits = self.classifier(temporal_features)
        return logits