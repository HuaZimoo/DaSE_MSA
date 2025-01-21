#!/bin/bash

python experiments/train.py \
    --no_weighted_loss \
    --no_balanced_sampler \
    --no_augmentation \
    --exp_name baseline_cv 