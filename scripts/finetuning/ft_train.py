from constants import RAW_DATA_FILEPATH, PROCESSED_DATA_DIR, METRICS_SAVE_DIR, ADAPTER_SAVE_DIR, SEED, FINETUNE_MODEL_NAME
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, Trainer, TrainingArguments
import os
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, List
from utils import set_seed
from scripts.finetuning.ft_utils import load_model_and_tokenizer, get_tunable_model, preprocess_data, define_optimizer, get_tokenized_datasets, gpu_info, get_args
from tqdm import tqdm
import json

def main():
    
    set_seed(SEED)
    gpu_info()
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # define paths
    project_path = os.getcwd()
    data_path = os.path.join(project_path, RAW_DATA_FILEPATH)
    save_data_path = os.path.join(project_path, PROCESSED_DATA_DIR)
    base_model_path = os.path.join(project_path, '..', FINETUNE_MODEL_NAME)
    metrics_save_path = os.path.join(project_path, METRICS_SAVE_DIR)
    adapter_save_path = os.path.join(project_path, ADAPTER_SAVE_DIR, f'{FINETUNE_MODEL_NAME}')
    print("debug 1")
    # load model and tokenizer     
    model, tokenizer = load_model_and_tokenizer(base_model_path)
    model = get_tunable_model(model,device)
    print("model loaded")
    # preprocess and save data, load tokenized data
    preprocess_data(data_path, save_data_path, max_samples=args.max_samples, train_ratio=0.5)
    train_dataset, val_dataset, _ = get_tokenized_datasets(save_data_path, tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )
    dev_dataloader = DataLoader(
        val_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )
  
    args.num_training_steps = len(train_dataloader) * args.max_epochs
    
    print("data loaders active")

    optimizer, lr_scheduler = define_optimizer(args, model)
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
    
    print('begin training!')

    for epoch in range(args.max_epochs):
        print(f'epoch {epoch}/{args.max_epochs}')
        trn_losses = []
        with tqdm(total=len(train_dataloader)) as pbar:
            for batch in train_dataloader:
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
                    detached_val_loss = outputs_val.loss.detach().float()
                    val_losses.append(detached_val_loss)
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

        print(f"epoch: {epoch}/{args.max_epochs}, "
            f"trn_loss_epoch: {trn_loss_epoch}, "
            f"trn_perplexity_epoch: {trn_perplexity_epoch}, "
            f"val_loss_epoch: {val_loss_epoch}, "
            f"val_perplexity_epoch: {val_perplexity_epoch}, "
            f"lr: {lr_scheduler.get_lr()[0]}")

        # Save metrics to JSON file after each epoch
        model_epoch_metrics_path = os.path.join(metrics_save_path, f'{FINETUNE_MODEL_NAME}', f'{epoch}')
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
            # Save the LoRA adapter weights
            if not os.path.exists(adapter_save_path):
                os.makedirs(adapter_save_path)
            model.save_pretrained(adapter_save_path)
            print(f'saved adapter to {adapter_save_path} at epoch {epoch}')
        


if __name__ == '__main__':
    main()