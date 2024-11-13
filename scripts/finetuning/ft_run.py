from constants import ZS_SYSTEM_PROMPT, API_KEY, LLM_NAME
import torch
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import peft
from transformers import default_data_collator
import argparse
from tqdm import tqdm
from scripts.finetuning.ft_utils import load_model_and_tokenizer, get_tunable_model, preprocess_data, define_optimizer
def main():
    
    # Check number of GPUs available
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    # If you want more details about each GPU
    if gpu_count > 0:
        for i in range(gpu_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            # Get memory info in bytes
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            print(f"Memory Allocated: {memory_allocated/1024**2:.2f}MB")
            print(f"Memory Reserved: {memory_reserved/1024**2:.2f}MB")
    
    # define paths
    project_path = os.getcwd()
    data_path = os.path.join(project_path, '..', '..', "data/raw/Hospitalization-Summarization.json")
    save_data_path = os.path.join(project_path, '..', '..', 'data/processed')
    model_name = 'Llama-3.2-1B'
    model_path = os.path.join(project_path, '..', '..', '..', model_name)
    metrics_save_path = os.path.join(project_path, '..', '..', 'results/metrics')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # clear device cache
    torch.cuda.empty_cache()
    # define args
    args = argparse.Namespace()
    args.device = device
    args.batch_size = 1
    args.max_training_epochs = 2
    args.patience = 2
    args.lr0 = 1e-3
    args.lr_schedule = 'linear_decay'
    args.lr_num_warmup_steps = 1
    args.gradient_accumulation_steps = 4    
    
    # load model and tokenizer to device    
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = get_tunable_model(model, args.device)
    preprocess_data(data_path, save_data_path, max_samples=16, split_ratio=0.5)

    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=2, pin_memory=True
    )
    dev_dataloader = DataLoader(
        val_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=2, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=2, pin_memory=True
    )
    args.num_training_steps = len(train_dataloader) * args.max_training_epochs


    optimizer, lr_scheduler = define_optimizer(args, model)
    
    #print args
    for i, arg in enumerate(args):
        print(f"Args[{i}]: {arg}")
        
    # At the beginning of your script, create a dictionary to store metrics
    metrics = {
        'train_loss': [],
        'train_perplexity': [],
        'val_loss': [],
        'val_perplexity': [],
        'learning_rate': [],
            'epoch': []
            }

    # logging with dictionary updates
    # In the training loop:
    model.train()
    best_val_loss = float('inf')
    patience = args.patience
    n_steps = 0
    trn_losses = []
    print('begin training!')

    for epoch in range(args.max_training_epochs):
        print(f'epoch {epoch}/{args.max_training_epochs}')
        with tqdm(total=len(train_dataloader)) as pbar:
            for idx_b, batch in enumerate(train_dataloader):
                n_steps += 1
                
                # forward pass 
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                
                # compute loss, gradient step 
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (n_steps % args.gradient_accumulation_steps == 0) or (n_steps == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
                
                lr_scheduler.step()
                
                detached_loss = loss.detach().float()
                trn_losses.append(detached_loss)
                
                # Store step-level metrics
                metrics['train_loss'].append(float(detached_loss))
                metrics['train_perplexity'].append(float(torch.exp(detached_loss)))
                pbar.update(1)
        
        # Validation loop remains same but store metrics
        val_losses = []
        with tqdm(total=len(dev_dataloader)) as pbar:
            for batch in dev_dataloader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs_val = model(**batch)
                    val_losses.append(outputs_val.loss.detach().float())
                pbar.update(1)

        # Calculate epoch metrics
        trn_loss_epoch = sum(trn_losses) / len(trn_losses)
        val_loss_epoch = sum(val_losses) / len(val_losses)
        trn_perplexity_epoch = torch.exp(trn_loss_epoch)
        val_perplexity_epoch = torch.exp(val_loss_epoch)
        
        # Store epoch-level metrics
        metrics['epoch'].append(epoch)
        metrics['learning_rate'].append(float(lr_scheduler.get_lr()[0]))
        metrics['val_loss'].append(float(val_loss_epoch))
        metrics['val_perplexity'].append(float(val_perplexity_epoch))

        print(f"epoch: {epoch}/{args.max_training_epochs}, "
            f"trn_loss_epoch: {trn_loss_epoch}, "
            f"trn_perplexity_epoch: {trn_perplexity_epoch}, "
            f"val_loss_epoch: {val_loss_epoch}, "
            f"val_perplexity_epoch: {val_perplexity_epoch}, "
            f"lr: {lr_scheduler.get_lr()[0]}")

        # Save metrics to JSON file after each epoch
        model_epoch_metrics_path = os.path.join(metrics_save_path, f'{model_name}', f'{epoch}')
        if not os.path.exists(model_epoch_metrics_path):
            os.makedirs(model_epoch_metrics_path)
        with open(os.path.join(model_epoch_metrics_path, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # early stopping
        if val_loss_epoch > best_val_loss:
            if patience == 0:
                print(f'stopping early at epoch {epoch}!')
                break
            else:
                patience -= 1
        else:
            patience = args.patience
            best_val_loss = val_loss_epoch
        
    # clear device cache
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


