#!/bin/bash

# 实验1：简单拼接融合
python experiments/train.py \
    --fusion_type concat \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name fusion_concat_balanced_aug

# 实验2：交叉注意力融合
python experiments/train.py \
    --fusion_type attention \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name fusion_attention_balanced_aug

# 实验3：门控融合
python experiments/train.py \
    --fusion_type gated \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name fusion_gated_balanced_aug

# 实验4：双线性融合
python experiments/train.py \
    --fusion_type bilinear \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name fusion_bilinear_balanced_aug 