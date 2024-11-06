#! /bin/bash

mkdir -p data/raw data/processed
mkdir -p scripts/data_exploration
mkdir -p scripts/training
mkdir -p scripts/evaluation
mkdir -p scripts/error_analysis
mkdir models
mkdir -p results/metrics
mkdir -p results/figures
mkdir logs
mkdir config
mkdir tests

touch scripts/data_exploration/data_exploration.ipynb
touch scripts/training/train_model.py
touch scripts/training/utils.py
touch scripts/evaluation/evaluate_model.py
touch scripts/error_analysis/error_analysis.ipynb
touch results/metrics/evaluation_metrics.json
touch config/config.yaml
touch requirements.txt
touch README.md
touch .gitignore

# how to run the script
# chmod +x setup.sh
# ./setup.sh
