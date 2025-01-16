#!/bin/bash



# Define parameter lists
llm_names=("Llama-3.2-1B" "Llama-3.2-1B-Instruct")  # "Llama-3.2-1B" "Llama-3.2-1B-Instruct"
lr0s=(1.0e-4 1.0e-5) 
gas=(8 16)
lr_scheduler_types=("polynomial_decay") # "cosine" "linear" "polynomial_decay" for poly decay, be mindful of power parameter since if it's 1 then scheduler is linear
optimizer_types=("adamw") # "adopt" "adamw"
dataset_path="/home/bulut/repositories/discharge-summarization/data/toy_custom_split"
# calculate number of combinations
num_combinations=$((${#llm_names[@]} * ${#lr0s[@]} * ${#gas[@]} * ${#lr_scheduler_types[@]} * ${#optimizer_types[@]}))
# initialize configs array with all combinations
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

# for loop to iterate over num_combinations
for i in $(seq 0 $((num_combinations - 1))); do
    identifier=run_${i}
    config=config_${i}
    read llm lr0 ga lr_scheduler optimizer <<< "${configs[$i]}"
    # create a config file first
    cat <<EOF > config/${config}.yaml
finetune_config:
    llm_name: $llm
    method: finetune
    batch_size: 1   
    max_epochs: 1
    lr0: $lr0
    patience: 7
    gradient_accumulation_steps: $ga
    lr_scheduler_type: $lr_scheduler
    lr_warmup_steps_ratio: 0.10
    poly_decay_power: 0.50
    optimizer_type: $optimizer
    dataset_path: $dataset_path
EOF

    # create a pbs script
    cat <<EOF > .run_pbs_files/${identifier}.pbs
#!/bin/bash
### Job Name
#PBS -N discharge_summarization-pipeline
### Project code
#PBS -A discharge_summarization
### Maximum time this job can run before being killed (here, 1 day)
#PBS -l walltime=01:00:00:00
### Resource Request (must contain cpucore, memory, and gpu (even if requested amount is zero)
#PBS -l cpucore=1:memory=32gb:gpu=1
### Output Options (default is stdout_and_stderr)
#PBS -l outputMode=stdout_and_stderr

. /home/bulut/miniconda3/etc/profile.d/conda.sh

conda activate hfenv
printenv

export OMP_NUM_THREADS=2

# Run Python scripts
(
  CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python -m scripts.finetune.ft_train --config ${config} --identifier ${identifier} &&
  CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python -m scripts.finetune.ft_inference --config ${config} --identifier ${identifier} &&
  CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python -m scripts.evaluation.eval_run --config ${config} --identifier ${identifier}
) &
wait
EOF
done





