#!/bin/bash

python experiments/train.py \
    --fusion_type bilinear \
    --use_balanced_sampler \
    --use_augmentation \
    --use_image --use_text \
    --exp_name ablation_text_only_bilinear

# 2. 仅使用图片模态
python experiments/train.py \
    --fusion_type bilinear \
    --use_balanced_sampler \
    --use_augmentation \
    --use_image --no_text \
    --exp_name ablation_image_only_bilinear

# 3. 仅使用文本模态
python experiments/train.py \
    --fusion_type bilinear \
    --use_balanced_sampler \
    --use_augmentation \
    --no_image --use_text \
    --exp_name ablation_text_only_bilinear