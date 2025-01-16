#!/bin/bash
#SBATCH --job-name=qlora_scoring
#SBATCH --output=.hpclogs/output_%j_%a.out
#SBATCH --error=.hpclogs/error_%j_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --account=nlp

source ~/.bashrc
conda activate scoring

python -m results.eval_scoring