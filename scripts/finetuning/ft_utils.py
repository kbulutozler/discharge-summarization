from constants import ZS_SYSTEM_PROMPT, API_KEY, FINETUNE_MODEL_NAME
import torch
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import peft
from constants import SEED
import argparse
from tqdm import tqdm

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
    inputs = tokenizer(test_sample, return_tensors="pt")
    print("shape of input_ids", inputs["input_ids"].shape)
    outputs = model.generate(inputs["input_ids"],
                            tokenizer=tokenizer,
                            min_new_tokens=250,
                            max_new_tokens=500,
                            temperature=0.8,
                            top_p=0.9,
                            top_k=40,
                            repetition_penalty=1.2,
                                stop_strings=["||endoftext||"]
                                )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
def generate_summaries(model, tokenizer, df):
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
    # Generate summaries using the Together.AI API
    for i, trial in tqdm(df.iterrows(), total=df.shape[0], desc=f"Generating summaries with {FINETUNE_MODEL_NAME}"):
        text = trial["text"]
        generated_summaries.append(single_inference_local(model, tokenizer, text))
    df.loc[:, "generated_summary"] = generated_summaries
    # drop report column
    df = df.drop(columns=["text"])
    
    return df
    
def get_args():
    parser = argparse.ArgumentParser(description="Finetuning Script")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr0', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of training epochs')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--lr_schedule', type=str, default='linear_decay', help='Learning rate schedule')
    parser.add_argument('--lr_num_warmup_steps', type=int, default=100, help='Number of warmup steps for learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    
    return parser.parse_args()

def gpu_info():
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


def load_model_and_tokenizer(model_path):
    ''' load model and tokenizer '''

    # set quantization configs if using qlora
    quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

    # define model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token_id is None: # autoregressive models' pad token not set by default
        tokenizer.pad_token_id = tokenizer.eos_token_id 

    return model, tokenizer

def get_tunable_model(model, device):
    ''' prep model for param-efficient fine-tuning '''

    task_type = peft.TaskType.CAUSAL_LM
    
    # prepare for k-bit training
    model = peft.prepare_model_for_kbit_training(model) 
    
    # get peft configs based on architecture (task_type) and fine-tuning method
    config = peft.LoraConfig(task_type=task_type, inference_mode=False,
                                r=8, lora_alpha=32,
                                lora_dropout=0.1)

    # wrap model w peft configs
    model = peft.get_peft_model(model, config).to(device)
    model.print_trainable_parameters()

    return model


def preprocess_data(data_path, save_data_path, max_samples, train_ratio=0.5):
    """ load data from json file to pandas dataframe """
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    data = pd.DataFrame(data)
    if max_samples is not None:
        data = data[:max_samples]
    # remove columns except for instruct and answer
    data = data[['instruct', 'answer']]
    # remove Input:\n from beginning of instruct texts
    data['instruct'] = data['instruct'].apply(lambda x: x.split('Input:\n')[1] if 'Input:\n' in x else x)
    # change column names to text and target
    data.rename(columns={'instruct': 'text', 'answer': 'target'}, inplace=True)
    # inject ||startoftext|| and ||endoftext||
    data['target'] = data['target'].apply(lambda x: "||startoftext||" + x + " ||endoftext||")
    
    # First split data into train and temp 
    train_data, temp_data = train_test_split(data, test_size=train_ratio, random_state=SEED)
    # Split temp data into val and test equally, half of the remaining data
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)
    
    # save to csv
    train_data.to_csv(os.path.join(save_data_path, 'train.csv'))
    val_data.to_csv(os.path.join(save_data_path, 'val.csv'))
    test_data.to_csv(os.path.join(save_data_path, 'test.csv'))

def tokenize_function(examples, tokenizer, max_length):
    """
    Preprocess function for autoregressive models using left padding to fit max_length.
    final version:
        input_ids: padding + text + label + eos (no masking, full text and label)
        attention_mask: padding + text + label + eos (all 1s except for padding)
        labels: padding + label + eos (-100 masks padding and input tokens)
    """
     # Insert special tokens into inputs and targets
    batch_size = len(examples['text'])
    inputs = examples['text']
    targets = examples['target']

    # Tokenize inputs and targets without padding
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets, add_special_tokens=False)
    # print("0th model_inputs", model_inputs["input_ids"][0])
    # print("length of 0th model_inputs", len(model_inputs["input_ids"][0]))
    # print("0th labels", labels["input_ids"][0])
    # print("length of 0th labels", len(labels["input_ids"][0]))

    for i in range(batch_size): # for each example input-target pairs
        # Get tokenized input and label IDs
        sample_input_ids = model_inputs["input_ids"][i]
        sample_label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

        # Append labels to inputs
        model_inputs["input_ids"][i] = sample_input_ids + sample_label_input_ids

        # Create labels for loss computation (-100 masks input tokens)
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + sample_label_input_ids

        # Update attention mask
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    # print("0th model_inputs", model_inputs["input_ids"][0])
    # print("length of 0th model_inputs", len(model_inputs["input_ids"][0]))
    # print("0th labels", labels["input_ids"][0])
    # print("length of 0th labels", len(labels["input_ids"][0]))
    # print("0th attention mask", model_inputs["attention_mask"][0])
    # print("length of 0th attention mask", len(model_inputs["attention_mask"][0]))
    # handle padding from left side, to fit max_length
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        sample_label_input_ids = labels["input_ids"][i]
        # pad input ids
        model_inputs["input_ids"][i] = ([tokenizer.pad_token_id] * 
                                       (max_length - len(sample_input_ids)) + 
                                       sample_input_ids)
        # pad attention mask
        model_inputs["attention_mask"][i] = ([0] * (max_length - len(sample_input_ids)) +
                                            model_inputs["attention_mask"][i])
        # pad labels
        labels["input_ids"][i] = ([-100] * (max_length - len(sample_label_input_ids)) +
                                 sample_label_input_ids)
        # truncate to max_length, but im not sure if this is necessary since we already calculated max_length on combined input-target length + 1 for eos token
        model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
        model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][:max_length]
        labels["input_ids"][i] = labels["input_ids"][i][:max_length]
        # print("0th model_inputs", model_inputs["input_ids"][0])
        # print("length of 0th model_inputs", len(model_inputs["input_ids"][0]))
        # print("0th labels", labels["input_ids"][0])
        # print("length of 0th labels", len(labels["input_ids"][0]))
        # print("0th attention mask", model_inputs["attention_mask"][0])
        # print("length of 0th attention mask", len(model_inputs["attention_mask"][0]))
        

    # Add labels to model inputs
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def calculate_max_length(dataset, tokenizer):
    max_lengths = []
    
    # Calculate max length for each split in the dataset
    tokenized_inputs = [tokenizer(example['text'])['input_ids'] for example in dataset]
    tokenized_targets = [tokenizer(example['target'])['input_ids'] for example in dataset]

    # Get the maximum length from the lengths of each text-text label pair and add 1 for eos token
    combined_max_length = max([len(input_ids) + len(label_ids) 
                              for input_ids, label_ids in zip(tokenized_inputs, tokenized_targets)]) + 1
    print("combined_max_length", combined_max_length)
    return combined_max_length

def get_hf_dataset(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, val_dataset, test_dataset

def get_tokenized_datasets(data_path, tokenizer):
    train_dataset, val_dataset, test_dataset = get_hf_dataset(data_path)
    train_max_length = calculate_max_length(train_dataset, tokenizer)
    val_max_length = calculate_max_length(val_dataset, tokenizer)
    test_max_length = calculate_max_length(test_dataset, tokenizer)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer, max_length=train_max_length),
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer, max_length=val_max_length),
        batched=True,
        num_proc=1,
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on val dataset",
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer, max_length=test_max_length),
        batched=True,
        num_proc=1,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on test dataset",
    )

    return train_dataset, val_dataset, test_dataset

def define_optimizer(args, model):
    ''' given parameters
        define optimizer '''
    
    # extract learning rate params
    lr0 = args.lr0 # initial learning rate

    # define optimizer, lr_scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=lr0,
                                   no_deprecation_warning=True)

    str_ = f'using linear scheduler with lr0 {lr0},'
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_num_warmup_steps,
        num_training_steps=args.num_training_steps,
    )
        
    str_ += f' and {args.lr_num_warmup_steps} warm-up steps!' 
    print(str_)

    return optimizer, lr_scheduler
