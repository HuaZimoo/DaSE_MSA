#!/bin/bash

echo "Running baseline experiment..."
bash experiments/baseline_exp.sh

echo "Running single strategy experiments..."
bash experiments/single_strategy_exp.sh

echo "Running combined strategy experiments..."
bash experiments/combined_strategy_exp.sh 