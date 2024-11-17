#!/bin/bash
#SBATCH --job-name=qlora_experiments
#SBATCH --output=.hpclogs/output_%A.out
#SBATCH --error=.hpclogs/error_%A.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:volta:1
#SBATCH --account=nlp
#SBATCH --mail-user=kbozler@arizona.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cd /xdisk/bethard/kbozler/repositories/discharge-summarization
source ~/.bashrc
conda activate hfenv

identifier=${SLURM_JOB_ID}
config=config



# Get parameters for this task
llm="Llama-3.2-1B-Instruct"
lr0=2.0e-5
gas=8
echo "Running configuration: Model = $llm, Learning Rate = $lr0, Gradient Accumulation Steps = $gas"
cat <<EOF > config/${config}.yaml
finetune_config:
    llm_name: "$llm"
    method: "finetune"
    batch_size: 1
    max_epochs: 1
    lr0: $lr0
    patience_ratio: 0.10
    gradient_accumulation_steps: $gas
    lr_schedule: "linear_decay"
    lr_warmup_steps_ratio: 0.15
EOF

# Run your Python scripts
python -m scripts.finetune.ft_train --config ${config} --identifier ${identifier}
python -m scripts.finetune.ft_inference --config ${config} --identifier ${identifier}
python -m scripts.evaluation.eval_run --config ${config} --identifier ${identifier}
