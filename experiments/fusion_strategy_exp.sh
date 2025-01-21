#!/bin/bash

python experiments/train.py \
    --fusion_type concat \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name fusion_concat_balanced_aug

python experiments/train.py \
    --fusion_type attention \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name fusion_attention_balanced_aug

python experiments/train.py \
    --fusion_type gated \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name fusion_gated_balanced_aug

python experiments/train.py \
    --fusion_type bilinear \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name fusion_bilinear_balanced_aug 