import torch
import torch.nn as nn
from transformers import CLIPModel
from src.models.fusion import (
    ConcatFusion, CrossAttentionFusion, 
    GatedFusion, BilinearFusion, EnhancedConcatFusion
)

class MultimodalSentimentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(config.clip_model_name)
        
        if 'large' in config.clip_model_name.lower():
            self.feature_dim = 768
        else:
            self.feature_dim = 512
        
        if config.fusion_type == 'concat':
            self.fusion_layer = ConcatFusion(self.feature_dim, config.hidden_dim)
        elif config.fusion_type == 'attention':
            self.fusion_layer = CrossAttentionFusion(
                self.feature_dim, config.hidden_dim, config.fusion_heads
            )
        elif config.fusion_type == 'gated':
            self.fusion_layer = GatedFusion(self.feature_dim, config.hidden_dim)
        elif config.fusion_type == 'bilinear':
            self.fusion_layer = BilinearFusion(self.feature_dim, config.hidden_dim)

        for name, param in self.clip.named_parameters():
            param.requires_grad = False
            if any(f'visual.transformer.resblocks.{i}' in name for i in [7, 8, 9, 10, 11]):
                param.requires_grad = True
            if any(f'text.transformer.resblocks.{i}' in name for i in [7, 8, 9, 10, 11]):
                param.requires_grad = True
            if 'visual.proj' in name or 'text.proj' in name:
                param.requires_grad = True
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        
    def forward(self, image_features, text_features, use_image=True, use_text=True): 
        image_outputs = self.clip.get_image_features(image_features) if use_image else None
        text_outputs = self.clip.get_text_features(text_features) if use_text else None
        
        if use_image and use_text:
            fused_features = self.fusion_layer(image_outputs, text_outputs)
        elif use_image:
            fused_features = image_outputs
        elif use_text:
            fused_features = text_outputs
        else:
            raise ValueError("At least one of use_image or use_text must be True.")
        
        logits = self.classifier(fused_features)
        return logits