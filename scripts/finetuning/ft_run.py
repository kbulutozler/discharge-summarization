from constants import ZS_SYSTEM_PROMPT, API_KEY, LLM_NAME, RAW_DATA_FILEPATH, PROCESSED_DATA_DIR
import torch
import os
import json
from torch.utils.data import DataLoader
from transformers import default_data_collator
import argparse
from tqdm import tqdm
from scripts.finetuning.ft_utils import load_model_and_tokenizer, get_tunable_model, preprocess_data, define_optimizer, get_tokenized_datasets, gpu_info, get_args
from utils import set_seed
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs



def main():
    """
    finetuning script for discharge summarization
    example command: 
    CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi_gpu --mixed_precision fp16 --module scripts.finetuning.ft_run --batch_size 1 --lr0 1e-3 --max_epochs 2 --patience 2 --model_name Llama-3.2-1B --lr_num_warmup_steps 2 --gradient_accumulation_steps 4
    """
    seed = 31
    set_seed(seed)
    gpu_info()
    args = get_args()
    
    # define paths
    project_path = os.getcwd()
    data_path = os.path.join(project_path, RAW_DATA_FILEPATH)
    save_data_path = os.path.join(project_path, PROCESSED_DATA_DIR)
    model_name = args.model_name
    model_path = os.path.join(project_path, '..',  model_name)
    metrics_save_path = os.path.join(project_path, 'results/metrics')
    print("debug 1")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    print("debug 2")
    
    # load model and tokenizer     
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = get_tunable_model(model)
    print("model loaded")
    # preprocess and save data, load tokenized data
    preprocess_data(data_path, save_data_path, max_samples=48, split_ratio=0.5)
    train_dataset, val_dataset, test_dataset = get_tokenized_datasets(save_data_path, tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )
    dev_dataloader = DataLoader(
        val_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )      
    args.num_training_steps = len(train_dataloader) * args.max_epochs
    
    print("data loaders active")

    optimizer, lr_scheduler = define_optimizer(args, model)
    
    # prepare everything
    print("debug 3")
    model, train_dataloader, dev_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, dev_dataloader, test_dataloader, optimizer, lr_scheduler
    )
    print('debug 4')
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Rank {accelerator.local_process_index} has {total_params} parameters.")
    print("args: ", args)
        
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
    

    for epoch in range(args.max_epochs):
        print(f'epoch {epoch}/{args.max_epochs}')
        with tqdm(total=len(train_dataloader)) as pbar:
            for batch in train_dataloader:
                n_steps += 1
                with accelerator.accumulate(model):
                    # forward pass 
                    outputs = model(**batch)
                    
                    # compute loss, gradient step 
                    loss = outputs.loss 
                    accelerator.backward(loss)
                    # right here you can use clip_grad_norm_
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
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
                with accelerator.accumulate(model):
                    # forward pass 
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


