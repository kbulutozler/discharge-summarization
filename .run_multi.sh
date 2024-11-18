#!/bin/bash
#SBATCH --job-name=qlora_experiments
#SBATCH --output=.hpclogs/output_%j_%a.out
#SBATCH --error=.hpclogs/error_%j_%a.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-3
#SBATCH --account=nlp
#SBATCH --mail-user=kbozler@arizona.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate hfenv

identifier=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
config=config_${SLURM_ARRAY_TASK_ID}

# Define parameter lists
llm_names=("Llama-3.2-3B" "Llama-3.2-3B-Instruct") # "Llama-3.2-1B-Instruct" 
lr0s=(1.0e-3) # 5.0e-3
gas=(16)
lr_scheduler_types=("polynomial_decay" "linear") # "cosine" "linear" "polynomial_decay" for poly decay, be mindful of power parameter since if it's 1 then scheduler is linear
optimizer_types=("adamw") # "adopt" "adamw"
dataset_path="/xdisk/bethard/kbozler/repositories/discharge-summarization/data/custom_split"
# Build combinations
declare -A configs
index=0
for llm in "${llm_names[@]}"; do
    for lr in "${lr0s[@]}"; do
        for ga in "${gas[@]}"; do
            for lr_scheduler in "${lr_scheduler_types[@]}"; do
                for optimizer in "${optimizer_types[@]}"; do
                    configs[$index]="$llm $lr $ga $lr_scheduler $optimizer"
                    ((index++))
                done
            done
        done
    done
done

# Read values for current task
read llm lr0 ga lr_scheduler optimizer <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

echo "Running configuration: Model = $llm, Learning Rate = $lr0, Gradient Accumulation Steps = $ga, Learning Rate Scheduler = $lr_scheduler, Optimizer = $optimizer, Dataset Path = $dataset_path"
cat <<EOF > config/${config}.yaml
finetune_config:
    llm_name: $llm
    method: finetune
    batch_size: 1   
    max_epochs: 8
    lr0: $lr0
    patience: 10
    gradient_accumulation_steps: $ga
    lr_scheduler_type: $lr_scheduler
    lr_warmup_steps_ratio: 0.10
    poly_decay_power: 0.75 
    optimizer_type: $optimizer
    dataset_path: $dataset_path
EOF

# Run your Python scripts
python -m scripts.finetune.ft_train --config ${config} --identifier ${identifier}
python -m scripts.finetune.ft_inference --config ${config} --identifier ${identifier}
python -m scripts.evaluation.eval_run --config ${config} --identifier ${identifier}
