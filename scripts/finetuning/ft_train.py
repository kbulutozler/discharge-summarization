import random
import string
from constants import PROCESSED_DATA_DIR, TRAINING_METRICS_SAVE_DIR, ADAPTER_SAVE_DIR, SEED, LOCAL_MODELS_DIR
from transformers import default_data_collator   
import os
import torch
from torch.utils.data import DataLoader
from utils import set_seed
from scripts.finetuning.ft_utils import load_model_and_tokenizer, get_lora_model, define_optimizer, from_df_to_tokenized_dataset, gpu_info, get_args, plot_losses
from tqdm import tqdm
import json
from torch.amp import autocast, GradScaler
import math



def main():
    """
    Finetuning Script for Language Models

    This script is designed to fine-tune a pre-trained language model using a specified dataset. 
    It allows for customization of various training parameters such as batch size, learning rate, 
    number of epochs, and early stopping criteria. The script also supports gradient accumulation 
    to effectively manage memory usage during training.

    Arguments:
    --batch_size: int, default=4
        The number of samples processed before the model is updated.
    --llm_name: str, default=None
        The name of the local model to finetune.
    --lr0: float, default=1e-3
        The initial learning rate for the optimizer.
    --max_epochs: int, default=5
        The maximum number of epochs to train the model.
    --patience: int, default=3
        The number of epochs with no improvement after which training will be stopped.

    --gradient_accumulation_steps: int, default=4
        The number of steps to accumulate gradients before updating the model parameters.
    """

    set_seed(SEED)
    gpu_info()
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    scaler = GradScaler(enabled=True)
    # define paths
    project_path = os.getcwd()
    processed_data_path = os.path.join(project_path, PROCESSED_DATA_DIR)
    base_model_path = os.path.join(LOCAL_MODELS_DIR, args.llm_name)
    #metrics_save_path = os.path.join(project_path, TRAINING_METRICS_SAVE_DIR, f'{args.llm_name}')
    adapter_save_path = os.path.join(project_path, ADAPTER_SAVE_DIR, f'{args.llm_name}')

    # load quantized model with lora and tokenizer     
    model, tokenizer = load_model_and_tokenizer(base_model_path)
    model = get_lora_model(model)
    model.to(args.device)
    model.gradient_checkpointing_enable()
    #### 
    # when batch size is 1 and gradient accumulation step is 2 and
    # with this configuration of the model, 4.2gb vram is used
    # batch=1, training begins with 19gb vram usage
    # in same loop, with gradient accumulation step is 2, which makes the vram to go up to 24gb every 2 steps
    # and then goes down to 19gb after optimizer step and optimizer zero grad



        
    dataset = from_df_to_tokenized_dataset(processed_data_path, tokenizer)
    adapter_save_path = os.path.join(adapter_save_path, f'split_{len(dataset["train"])}_{len(dataset["validation"])}')
    train_dataloader = DataLoader(
        dataset['train'], shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )
    dev_dataloader = DataLoader(
        dataset['validation'], shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size
    )
    
    args.num_training_steps = math.ceil(len(train_dataloader) * args.max_epochs)
    args.lr_num_warmup_steps = math.ceil(args.num_training_steps * 0.15)
    
    print(f"when batch size is {args.batch_size}, num training steps is {args.num_training_steps} "
            f"because total number of train samples is {len(dataset['train'])} and maximum number of epochs is {args.max_epochs}")
    
    optimizer, lr_scheduler = define_optimizer(args, model)
    # create a dictionary to store metrics
    metrics = {
        'train_loss_per_step': [],
        'val_loss_per_step': [],
        'learning_rate': [],
        'epoch': [],
        'train_loss_per_epoch': [],
        'val_loss_per_epoch': [],
        
        }

    # logging with dictionary updates
    
    best_val_loss = float('inf')
    patience = args.patience
    n_steps = 0
    
    print('begin training!')


    for epoch in range(args.max_epochs):
        print(f'epoch {epoch+1}/{args.max_epochs}')
        epoch_trn_losses = []
        model.train()
        
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
                n_steps += 1
                
                # forward pass
                batch = {k: v.to(args.device) for k, v in batch.items()}
                with autocast(device_type=device.type): # enable mixed precision training
                    outputs = model(**batch)
                    # compute loss
                    loss = outputs.loss / args.gradient_accumulation_steps # outputs.loss is averaged over the batch already, here it gets divided again by gradient accumulation steps
            
                # backward pass
                scaler.scale(loss).backward() # enable gradient scaling for mixed precision training
                
                # Update weights after gradient accumulation steps or at epoch end
                if (n_steps % args.gradient_accumulation_steps == 0) or (n_steps == len(train_dataloader)):
                    scaler.step(optimizer) # update weights
                    scaler.update() # update scaling factor. scaling factor is used to scale the loss value in a way that prevents overflow or underflow during backpropagation
                    optimizer.zero_grad() # reset the gradients
                    
                    # Update learning rate
                    lr_scheduler.step()
                
                # Detach loss for logging
                detached_loss = loss.detach().item()
                epoch_trn_losses.append(detached_loss)
                if (n_steps % args.gradient_accumulation_steps == 0) or (n_steps == len(train_dataloader)):
                    del outputs, loss # delete variables to free up memory
                    torch.cuda.empty_cache() # clear the cache to free up memory. cache includes intermediate tensors and gradients that are no longer needed
                del batch # vram goes down to 19gb but then quicly goes up to 24 gb because batch is loaded again in the training loop! and all the forward pass loss calculation grad calculation takes longer. that's why vram stays at 24gb longer. (gradient accumulation steps is 8 btw)
                # Store step-level metric
                if n_steps % int(len(train_dataloader)/len(dev_dataloader)) == 0:
                    metrics['train_loss_per_step'].append(float(detached_loss))
        
        # Validation loop remains same but store metrics
        epoch_val_losses = []
        model.eval()
        for batch in tqdm(dev_dataloader, total=len(dev_dataloader), desc="Validation"):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                with torch.no_grad():
                    # forward pass
                    outputs_val = model(**batch)
                    # compute loss
                    val_loss = outputs_val.loss 
                    # detach loss for logging
                    detached_val_loss = val_loss.detach().item()
                    epoch_val_losses.append(detached_val_loss)
                    # Store step-level metric
                    metrics['val_loss_per_step'].append(float(detached_val_loss))
                    del batch, outputs_val, val_loss

        # Calculate epoch-level metrics
        trn_loss_epoch = sum(epoch_trn_losses) / len(epoch_trn_losses)
        val_loss_epoch = sum(epoch_val_losses) / len(epoch_val_losses)
        
        # Store epoch-level metrics
        metrics['epoch'].append(epoch+1)
        metrics['learning_rate'].append(float(lr_scheduler.get_last_lr()[0]))
        metrics['train_loss_per_epoch'].append(float(trn_loss_epoch))
        metrics['val_loss_per_epoch'].append(float(val_loss_epoch))
        
        # print epoch level metrics
        print(f'epoch {epoch+1} metrics:')
        print(f'train loss: {trn_loss_epoch}, val loss: {val_loss_epoch}, learning rate: {lr_scheduler.get_last_lr()[0]}')

        if val_loss_epoch < best_val_loss:
            print(f"current val loss {val_loss_epoch} is less than best val loss {best_val_loss}")
            best_val_loss = val_loss_epoch
            # Save the LoRA adapter weights when you get best val loss, in the end there will be 1 saved adapter that has the lowest val loss
            if not os.path.exists(adapter_save_path):
                os.makedirs(adapter_save_path)
            model.save_pretrained(adapter_save_path)
            print(f'saved adapter to {adapter_save_path} at epoch {epoch+1}')
            
        del epoch_trn_losses, epoch_val_losses
            
    # Save metrics to JSON file after training
    plot_losses(metrics['train_loss_per_step'], metrics['val_loss_per_step'], os.path.join(adapter_save_path, 'loss_plot.png'))
    if not os.path.exists(adapter_save_path):
        os.makedirs(adapter_save_path)
    with open(os.path.join(adapter_save_path, f'training_log.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()