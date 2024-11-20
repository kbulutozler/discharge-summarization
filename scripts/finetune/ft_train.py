import random
import string
from constants import SEED, LOCAL_MODELS_PATH, CUSTOM_SPLIT_PATH, OUTPUT_MODEL_PATH, RUN_ARGS_PATH
from transformers import default_data_collator   
import os
import torch
from torch.utils.data import DataLoader
from utils import set_seed, get_args
from scripts.finetune.ft_utils import load_model_and_tokenizer, get_lora_model, define_optimizer, from_df_to_tokenized_dataset, gpu_info, save_args_to_json, plot_training_metrics, calc_num_learning_steps
from tqdm import tqdm
import json
from torch.amp import autocast, GradScaler
import math
import uuid

def main():
    """
    Finetuning Script for Language Models

    This script is designed to fine-tune a pre-trained language model using a specified dataset. 
    It allows for customization of various training parameters such as batch size, learning rate, 
    number of epochs, and early stopping criteria. The script also supports gradient accumulation 
    to effectively manage memory usage during training.

    Arguments:
    

    """

    set_seed(SEED)
    gpu_info()
    args = get_args("finetune_config")
    identifier = args.identifier # identifies the run, hparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=True)
    # define paths
    base_model_path = os.path.join(LOCAL_MODELS_PATH, args.llm_name)
    model_save_path = os.path.join(OUTPUT_MODEL_PATH, f'{args.llm_name}', identifier)
    os.makedirs(model_save_path, exist_ok=True)

    # load quantized model with lora and tokenizer     
    model, tokenizer = load_model_and_tokenizer(base_model_path)
    model = get_lora_model(model)
    model.to(device)
    model.gradient_checkpointing_enable()
        
    dataset = from_df_to_tokenized_dataset(args.dataset_path, tokenizer)
    args.train_samples = len(dataset['train'])
    args.dev_samples = len(dataset['dev'])
    train_dataloader = DataLoader(
        dataset['train'], shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )
    dev_dataloader = DataLoader(
        dataset['dev'], shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )
    eval_frequency = args.gradient_accumulation_steps * 2  # Evaluation frequency. if gradient accumulation steps is 4, then evaluate every 8 steps
    num_batches_per_epoch = len(train_dataloader)
    args.num_training_steps = num_batches_per_epoch * args.max_epochs # total number of forward/backward passes, NOT the number of optimizer updates bc of gradient accumulation
    args.num_learning_steps = calc_num_learning_steps(num_batches_per_epoch, args.gradient_accumulation_steps, args.max_epochs) # number of times model weights are updated
    args.lr_num_warmup_steps = math.floor(args.num_learning_steps * args.lr_warmup_steps_ratio) # basically number of times lr is updated * warmup ratio
    print(f"when batch size is {args.batch_size}, num training steps is {args.num_training_steps} "
            f"because number of train samples is {args.train_samples} and maximum number of epochs is {args.max_epochs}")
    
    optimizer, lr_scheduler = define_optimizer(args, model)
    # create a dictionary to store metrics
    metrics = {
        'train_loss_per_eval_step': [],
        'val_loss_per_eval_step': [],
        'learning_rate_per_eval_step': [],
        'learning_rate_per_epoch': [],
        'epoch': [],
        'train_loss_per_epoch': [],
        'val_loss_per_epoch': [],
    }

    # Initialize variables for tracking losses and steps
    best_val_loss = float('inf')
    n_steps = 0
    
    trn_losses_since_eval = []
    eval_steps = []
    patience = args.patience
    print('begin training!')

    for epoch in range(args.max_epochs):
        print(f'epoch {epoch+1}/{args.max_epochs}')
        epoch_trn_losses = []
        model.train()
        
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
            n_steps += 1
            
            # forward pass
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type=device.type):  # enable mixed precision training
                outputs = model(**batch)
                # compute loss
                loss = outputs.loss / args.gradient_accumulation_steps
            
            # backward pass
            scaler.scale(loss).backward()  # enable gradient scaling for mixed precision training
            
            # Detach loss for logging
            detached_loss = loss.detach().item()
            trn_losses_since_eval.append(detached_loss)
            epoch_trn_losses.append(detached_loss)
            
            # Update weights after gradient accumulation steps or at epoch end
            if (n_steps % args.gradient_accumulation_steps == 0) or (n_steps % len(train_dataloader) == 0):
                print(f'update weights and learning rate at step {n_steps}')
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                del outputs, loss  # free up memory
                torch.cuda.empty_cache()
            
            del batch  # free up memory

            # Perform evaluation every 'eval_frequency' steps or at the end of the epoch
            if n_steps % eval_frequency == 0 or n_steps % len(train_dataloader) == 0:
                print(f'evaluate at step {n_steps}')
                eval_steps.append(n_steps)
                # Compute average training loss since last evaluation
                trn_loss_avg = sum(trn_losses_since_eval) / len(trn_losses_since_eval)
                metrics['train_loss_per_eval_step'].append(float(trn_loss_avg))
                trn_losses_since_eval = []  # reset for next interval
                metrics['learning_rate_per_eval_step'].append(float(lr_scheduler.get_last_lr()[0]))

                # Evaluation loop
                model.eval()
                val_losses = []
                for val_batch in tqdm(dev_dataloader, total=len(dev_dataloader), desc="Validation"):
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    with torch.no_grad():
                        outputs_val = model(**val_batch)
                        val_loss = outputs_val.loss
                        detached_val_loss = val_loss.detach().item()
                        val_losses.append(detached_val_loss)
                    del val_batch, outputs_val, val_loss
                val_loss_avg = sum(val_losses) / len(val_losses)
                metrics['val_loss_per_eval_step'].append(float(val_loss_avg))
                # Print or log the training and validation loss
                print(f"Step {n_steps}: Train loss {trn_loss_avg}, Val loss {val_loss_avg}")

                # Check if current validation loss is the best
                if val_loss_avg < best_val_loss:
                    patience = args.patience
                    print(f"current val loss {val_loss_avg} is less than best val loss {best_val_loss}")
                    best_val_loss = val_loss_avg
                    args.best_val_loss = best_val_loss
                    args.best_val_loss_step = n_steps
                    # Save the LoRA adapter weights
                    model.save_pretrained(model_save_path)
                    print(f'saved model to {model_save_path} at step {n_steps}')
                else:
                    patience -= 1
                    print(f'current val loss {val_loss_avg} is not less than best val loss {best_val_loss}, reducing patience by 1, to {patience}')
                    if patience == 0:
                        print(f"early stopping at epoch {epoch+1}")
                        break
                model.train()
        if patience == 0:
            break
                
            
        

        # End of epoch metrics
        trn_loss_epoch = sum(epoch_trn_losses) / len(epoch_trn_losses)
        metrics['epoch'].append(epoch+1)
        metrics['learning_rate_per_epoch'].append(float(lr_scheduler.get_last_lr()[0]))
        metrics['train_loss_per_epoch'].append(float(trn_loss_epoch))
        # Use the last validation loss average for epoch-level metric
        metrics['val_loss_per_epoch'].append(float(val_loss_avg))
        # Print epoch level metrics
        print(f'epoch {epoch+1} metrics:')
        print(f'train loss: {trn_loss_epoch}, val loss: {val_loss_avg}, learning rate: {lr_scheduler.get_last_lr()[0]}')

    # Save metrics and plot after training
    plot_training_metrics(args, metrics, eval_steps, model_save_path)
    with open(os.path.join(model_save_path, f'training_log.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    # save args
    save_args_to_json(args.__dict__, identifier, os.path.join(RUN_ARGS_PATH, f"{identifier}.json"))
    save_args_to_json(args.__dict__, identifier, os.path.join(model_save_path, f"run_args.json"))
    print(f"finished training for {identifier}")

if __name__ == '__main__':
    main()