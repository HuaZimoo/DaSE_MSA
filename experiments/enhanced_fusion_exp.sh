#!/bin/bash

# 实验1：增强型拼接融合
python experiments/train.py \
    --fusion_type enhanced_concat \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name fusion_enhanced_concat

# 实验2：优化的CLIP特征
python experiments/train.py \
    --fusion_type concat \
    --feature_adaptation \
    --unfreeze_layers 2 \
    --exp_name optimized_clip_features 