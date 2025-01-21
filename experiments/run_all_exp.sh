#!/bin/bash

echo "Running baseline experiment..."
bash experiments/baseline_exp.sh

echo "Running single strategy experiments..."
bash experiments/single_strategy_exp.sh

echo "Running fusion strategy experiments..."
bash experiments/fusion_strategy_exp.sh

echo "Running modality ablation experiments..."
bash experiments/modality_ablation.sh