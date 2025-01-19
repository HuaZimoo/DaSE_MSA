#!/bin/bash

python experiments/train.py \
    --no_weighted_loss \
    --use_balanced_sampler \
    --use_augmentation \
    --exp_name sampler_aug_cv