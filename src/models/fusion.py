import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    """简单拼接融合"""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, image_features, text_features):
        # 确保特征维度匹配
        if image_features.dim() != text_features.dim():
            raise ValueError(f"Feature dimensions don't match: image={image_features.shape}, text={text_features.shape}")
        combined = torch.cat([image_features, text_features], dim=1)
        return self.fusion(combined)

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合"""
    def __init__(self, feature_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, image_features, text_features):
        # 添加序列维度
        img_seq = image_features.unsqueeze(0)
        txt_seq = text_features.unsqueeze(0)
        
        # 交叉注意力
        attn_output, _ = self.attention(img_seq, txt_seq, txt_seq)
        
        # 融合
        combined = torch.cat([attn_output.squeeze(0), text_features], dim=1)
        return self.fusion(combined)

class GatedFusion(nn.Module):
    """门控融合"""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, image_features, text_features):
        # 计算门控权重
        gate = self.gate(torch.cat([image_features, text_features], dim=1))
        # 加权融合
        fused = gate * image_features + (1 - gate) * text_features
        return self.fusion(fused)

class BilinearFusion(nn.Module):
    """双线性融合"""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(feature_dim, feature_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, image_features, text_features):
        fused = self.bilinear(image_features, text_features)
        return self.fusion(fused)

class EnhancedConcatFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = config.hidden_dim
        
        # 轻量级交叉注意力
        self.light_attention = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征增强层
        self.enhancement = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.ReLU()
        )
    
    def forward(self, image_features, text_features):
        # 基础拼接
        concat_features = torch.cat([image_features, text_features], dim=1)
        
        # 计算交互权重
        interaction_weight = self.light_attention(concat_features)
        
        # 特征增强
        enhanced_features = self.enhancement(concat_features)
        
        # 加权组合
        final_features = concat_features + interaction_weight * enhanced_features
        
        return final_features 