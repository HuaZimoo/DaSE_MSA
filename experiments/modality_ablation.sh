#!/bin/bash

python experiments/train.py \
    --fusion_type bilinear \
    --use_balanced_sampler \
    --use_augmentation \
    --use_image --use_text \
    --exp_name ablation_both_bilinear

python experiments/train.py \
    --fusion_type bilinear \
    --use_balanced_sampler \
    --use_augmentation \
    --use_image --no_text \
    --exp_name ablation_image_only_bilinear

python experiments/train.py \
    --fusion_type bilinear \
    --use_balanced_sampler \
    --use_augmentation \
    --no_image --use_text \
    --exp_name ablation_text_only_bilinear