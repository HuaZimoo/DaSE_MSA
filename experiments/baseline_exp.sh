#!/bin/bash

# 基线实验（交叉验证）
python experiments/train.py \
    --no_weighted_loss \
    --no_balanced_sampler \
    --no_augmentation \
    --exp_name baseline_cv 