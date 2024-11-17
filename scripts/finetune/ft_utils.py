import torch
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import CONFIG_PATH
import transformers
import peft
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import yaml
from adopt import ADOPT

def save_args_to_json(args_dict, identifier, save_path):
    """
    Save args dictionary to a JSON file. If file exists, update it with new data.
    
    Args:
        args_dict (dict): Dictionary containing arguments
        identifier (str): Unique identifier for the run
        save_path (str): Path to save the JSON file
    """
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            json.dump({identifier: args_dict}, f, indent=4)
    else:
        with open(save_path, 'r') as f:
            existing_data = json.load(f)
        existing_data[identifier] = args_dict
        with open(save_path, 'w') as f:
            json.dump(existing_data, f, indent=4)

# INFERENCE HELPER METHODS
def single_inference_local(model, tokenizer, test_sample):
    """
    Generates a single summary using the local model
    Args:
        model (transformers.AutoModelForCausalLM): model to use
        test_sample (str): sample to generate a summary for

    Returns:
        str: generated summary
    """
    # generate summary
    device = model.device
    inputs = tokenizer(test_sample, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print("device", inputs["input_ids"].device)
    outputs = model.generate(inputs["input_ids"],
                            tokenizer=tokenizer,
                            min_new_tokens=250,
                            max_new_tokens=500,
                            temperature=0.8,
                            top_p=0.9,
                            top_k=40,
                            repetition_penalty=1.2,
                            stop_strings=["||endoftext||"]
                            ) #check stop strings class 
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
def generate_summaries(args, model, tokenizer, df):
    """
    Generates summaries from df
    Args:
        model (transformers.AutoModelForCausalLM): model to use
        tokenizer (transformers.AutoTokenizer): tokenizer to use
        df (pd.DataFrame): dataframe with discharge reports

    Returns:
        pd.DataFrame: dataframe with generated summaries
    """
    # add empty column for generated summaries  
    generated_summaries = []
    full_output = []
    for i, trial in tqdm(df.iterrows(), total=df.shape[0], desc=f"Generating summaries with {args.llm_name}"):
        if (i+1)/df.shape[0] % 0.1 == 0:
            print(f"test sample {i+1} of {df.shape[0]}")
        text = trial["text"]
        trial_summary = single_inference_local(model, tokenizer, text)
        full_output.append(trial_summary)
        generated_summaries.append(trial_summary[len(text):])
    df.loc[:, "generated_summary"] = generated_summaries
    df.loc[:, "report_and_generated_summary"] = full_output
    # drop report column
    df = df.drop(columns=["text"])
    
    return df

def gpu_info():
    # Check number of GPUs available
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    # If you want more details about each GPU
    if gpu_count > 0:
        for i in range(gpu_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")


def load_model_and_tokenizer(model_path):
    ''' load model and tokenizer '''

    # set quantization configs if using qlora
    quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

    # define model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def get_lora_model(model):
    ''' add peft adapter to model '''

    task_type = peft.TaskType.CAUSAL_LM
    
    # prepare for k-bit training
    model = peft.prepare_model_for_kbit_training(model) 
    
    # get peft configs based on architecture (task_type) and fine-tuning method
    config = peft.LoraConfig(   
                                task_type=task_type, 
                                inference_mode=False,
                                r=8, 
                                lora_alpha=32,
                                lora_dropout=0.1
                            )

    # wrap model with peft configs
    model = peft.get_peft_model(model, config)
    model.print_trainable_parameters()

    return model

def tokenize_function(example, tokenizer):
    inputs = tokenizer(example["text"], add_special_tokens=True) # only gonna add bos
    targets = tokenizer(example["target"], add_special_tokens=False) # we will manually add eos
    for i in range(len(inputs["input_ids"])):
        sample_input_ids =  inputs["input_ids"][i] 
        sample_label_input_ids = targets["input_ids"][i] + [tokenizer.eos_token_id]
        inputs["input_ids"][i] = sample_input_ids + sample_label_input_ids
        targets["input_ids"][i] = [-100] * len(sample_input_ids) + sample_label_input_ids
        inputs["attention_mask"][i] = [1] * len(inputs["input_ids"][i])
    inputs["labels"] = targets["input_ids"]
    # input_ids, attention_mask, and labels are all the same length for a given sample, but not across samples
    # so we need to pad to max length from left side

    max_length = max([len(x) for x in inputs["input_ids"]])
    # add padding tokens to the left side of the input ids, attention mask, and labels
    for i in range(len(inputs["input_ids"])):
        inputs["input_ids"][i] = ([tokenizer.pad_token_id] * 
                                (max_length - len(inputs["input_ids"][i])) + 
                                inputs["input_ids"][i])
        inputs["attention_mask"][i] = ([0] * (max_length - len(inputs["attention_mask"][i])) +
                                    inputs["attention_mask"][i])
        inputs["labels"][i] = ([-100] * (max_length - len(inputs["labels"][i])) +
                                inputs["labels"][i])
        
    return inputs


def from_df_to_hf_dataset(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(data_path, 'dev.csv'))
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    return DatasetDict({"train": train_dataset, "dev": dev_dataset})

def from_df_to_tokenized_dataset(data_path, tokenizer):
    dataset = from_df_to_hf_dataset(data_path)

    dataset = dataset.map( # batch should be none to ensure max length is calculated for all samples in a split
            lambda x: tokenize_function(x, tokenizer=tokenizer),
            batched=True,
            remove_columns=dataset['train'].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )
    return dataset

def define_optimizer(args, model):
    ''' given parameters
        define optimizer '''
    
    # extract learning rate params
    lr0 = args.lr0 # initial learning rate
    
    # define optimizer
    if args.optimizer_type == "adamw":
        optimizer = transformers.AdamW(model.parameters(), lr=lr0,
                                   no_deprecation_warning=True)
    elif args.optimizer_type == "adopt":
        optimizer = ADOPT(model.parameters(), lr=lr0, decoupled=True)
    else:
        raise ValueError(f"Unsupported optimizer_type: {args.optimizer_type}")

    if args.lr_scheduler_type == "linear":
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_num_warmup_steps,
            num_training_steps=args.num_training_steps,
        )
    elif args.lr_scheduler_type == "polynomial_decay":
        lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_num_warmup_steps,
            num_training_steps=args.num_training_steps,
            power=1.0
        )
    elif args.lr_scheduler_type == "cosine":
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_num_warmup_steps,
            num_training_steps=args.num_training_steps,
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler_type: {args.lr_scheduler_type}")

    print(f'Using {args.lr_scheduler_type} scheduler and {args.optimizer_type} optimizer with lr0 {lr0} and {args.lr_num_warmup_steps} warm-up steps!')

    return optimizer, lr_scheduler

def plot_losses(train_losses, valid_losses, eval_steps, save_path):
    """
    Plots training and validation losses on the same graph.

    Args:
        train_losses (list): List of training losses per step.
        valid_losses (list): List of validation losses per step.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(eval_steps, train_losses, label='Training Loss', color='blue', linestyle='-', linewidth=2)
    plt.plot(eval_steps, valid_losses, label='Validation Loss', color='orange', linestyle='--', linewidth=2)
    plt.title('Training and Validation Losses', fontsize=16)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


