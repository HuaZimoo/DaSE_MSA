#!/bin/bash

python experiments/train.py \
    --use_weighted_loss \
    --no_balanced_sampler \
    --no_augmentation \
    --exp_name weighted_loss_cv

python experiments/train.py \
    --no_weighted_loss \
    --use_balanced_sampler \
    --no_augmentation \
    --exp_name balanced_sampler_cv

python experiments/train.py \
    --no_weighted_loss \
    --no_balanced_sampler \
    --use_augmentation \
    --exp_name augmentation_cv 