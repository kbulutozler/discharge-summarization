#!/bin/bash
#SBATCH --job-name=qlora_experiments
#SBATCH --output=.hpclogs/output_%A_%a.out
#SBATCH --error=.hpclogs/error_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-7
#SBATCH --account=nlp
#SBATCH --mail-user=kbozler@arizona.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cd /xdisk/bethard/kbozler/repositories/discharge-summarization
source ~/.bashrc
conda activate hfenv

identifier=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
config=config_${SLURM_ARRAY_TASK_ID}

# Define parameter lists
llm_names=("Llama-3.2-1B" "Llama-3.2-1B-Instruct")
lr0s=(1.0e-5 2.0e-5)
gas=(8 16)

# Build combinations
declare -A configs
index=0
for llm in "${llm_names[@]}"; do
    for lr in "${lr0s[@]}"; do
        for ga in "${gas[@]}"; do
            configs[$index]="$llm $lr $ga"
            ((index++))
        done
    done
done

# Read values for current task
read llm lr0 ga <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

echo "Running configuration: Model = $llm, Learning Rate = $lr0, Gradient Accumulation Steps = $ga"
cat <<EOF > config/${config}.yaml
finetune_config:
    llm_name: "$llm"
    method: "finetune"
    batch_size: 1
    max_epochs: 8
    lr0: $lr0
    patience_ratio: 0.10
    gradient_accumulation_steps: $ga
    lr_schedule: "linear_decay"
    lr_warmup_steps_ratio: 0.15
EOF

# Run your Python scripts
python -m scripts.finetune.ft_train --config ${config} --identifier ${identifier}
python -m scripts.finetune.ft_inference --config ${config} --identifier ${identifier}
python -m scripts.evaluation.eval_run --config ${config} --identifier ${identifier}
