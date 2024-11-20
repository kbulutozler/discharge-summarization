#!/bin/bash
#SBATCH --job-name=qlora_experiments
#SBATCH --output=.hpclogs/SINGLE_output_%A.out
#SBATCH --error=.hpclogs/SINGLE_error_%A.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:volta:1
#SBATCH --account=nlp


source ~/.bashrc
conda activate hfenv

identifier=${SLURM_JOB_ID}
config=config



# Get parameters for this task
llm="Llama-3.2-1B-Instruct"
lr0=1.0e-4
ga=2
lr_scheduler="polynomial_decay"
optimizer="adopt"
dataset_path="/xdisk/bethard/kbozler/repositories/discharge-summarization/data/toy_custom_split"

echo "Running configuration: Model = $llm, Learning Rate = $lr0, Gradient Accumulation Steps = $gas, Learning Rate Scheduler = $lr_scheduler, Optimizer = $optimizer, Dataset Path = $dataset_path"
cat <<EOF > config/${config}.yaml
finetune_config:
    llm_name: $llm
    method: finetune
    batch_size: 1   
    max_epochs: 7
    lr0: $lr0
    patience: 5
    gradient_accumulation_steps: $ga
    lr_scheduler_type: $lr_scheduler
    lr_warmup_steps_ratio: 0.15
    poly_decay_power: 0.50
    optimizer_type: $optimizer
    dataset_path: $dataset_path
EOF

# Run your Python scripts
python -m scripts.finetune.ft_train --config ${config} --identifier ${identifier}
python -m scripts.finetune.ft_inference --config ${config} --identifier ${identifier}
python -m scripts.evaluation.eval_run --config ${config} --identifier ${identifier}
