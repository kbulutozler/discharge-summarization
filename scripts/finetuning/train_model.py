from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, Trainer, TrainingArguments
import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
import transformers
from typing import Any, Dict, List
import peft

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


def preprocess_function(examples, tokenizer, max_length, device):
    """
    Preprocess function for autoregressive models using left padding to fit max_length.
    final version:
        input_ids: padding + text + label + eos (no masking, full text and label)
        attention_mask: padding + text + label + eos (all 1s except for padding)
        labels: padding + label + eos (-100 masks padding and input tokens)
    """
     # Insert special tokens into inputs and targets
    inputs = examples['text']
    targets = examples['text_label']


    # Tokenize inputs and targets without padding
    model_inputs = tokenizer(inputs, add_special_tokens=False)
    labels = tokenizer(targets, add_special_tokens=False)

    for i in range(len(inputs)): # for each example input-target pairs
        # Get tokenized input and label IDs
        sample_input_ids = model_inputs["input_ids"][i]
        sample_label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

        # Append labels to inputs
        model_inputs["input_ids"][i] = sample_input_ids + sample_label_input_ids

        # Create labels for loss computation (-100 masks input tokens)
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + sample_label_input_ids

        # Update attention mask
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    # handle padding from left side, to fit max_length
    for i in range(len(inputs)):
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

    # Add labels to model inputs
    model_inputs["labels"] = labels["input_ids"]

    # truncate to max_length, but im not sure if this is necessary since we already calculated max_length on combined input-target length + 1 for eos token
    model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length],device=device)
    model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length],device=device)
    labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length],device=device)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def calculate_max_length(dataset, tokenizer):
    max_lengths = []
    
    # Calculate max length for each split in the dataset
    tokenized_inputs = [tokenizer(example['text'])['input_ids'] for example in dataset]
    tokenized_targets = [tokenizer(example['text_label'])['input_ids'] for example in dataset]

    # Get the maximum length from the lengths of each text-text label pair and add 1 for eos token
    combined_max_length = max([len(input_ids) + len(label_ids) 
                              for input_ids, label_ids in zip(tokenized_inputs, tokenized_targets)]) + 1
    return combined_max_length

def get_hf_dataset(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    train_df.columns = ['text', 'text_label']
    test_df.columns = ['text', 'text_label']
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset

def get_tokenized_datasets(data_path, tokenizer, device):
    train_dataset, test_dataset = get_hf_dataset(data_path)
    train_max_length = calculate_max_length(train_dataset, tokenizer)
    test_max_length = calculate_max_length(test_dataset, tokenizer)

    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer, max_length=train_max_length, device=device),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer, max_length=test_max_length, device=device),
        batched=True
    )

    return train_dataset, test_dataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # load model and tokenizer with qlora 
    model_path = "/home/kbozler/Documents/Llama-3.1-8B"
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = get_tunable_model(model, device)

    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data/processed")

    train_dataset, test_dataset = get_tokenized_datasets(data_path, tokenizer, device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-3,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="no",  # Disable intermediate model saving
        gradient_accumulation_steps=4,  # Set gradient accumulation steps
        lr_scheduler_type="linear",  # Set learning rate schedule to linear decay
        warmup_steps=100,  # Set number of warmup steps for the scheduler
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    # Train 
    trainer.train()    

if __name__ == '__main__':
    main()


