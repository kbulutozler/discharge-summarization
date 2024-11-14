import torch
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import peft
import argparse
from tqdm import tqdm


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
                            )
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
    # Generate summaries using the Together.AI API
    for i, trial in tqdm(df.iterrows(), total=df.shape[0], desc=f"Generating summaries with {args.llm_name}"):
        if (i+1)/df.shape[0] % 0.1 == 0:
            print(f"test sample {i}")
        text = trial["text"]
        generated_summaries.append(single_inference_local(model, tokenizer, text))
    df.loc[:, "generated_summary"] = generated_summaries
    # drop report column
    df = df.drop(columns=["text"])
    
    return df


# FINETUNING HELPER METHODS
def get_args():
    parser = argparse.ArgumentParser(description="Finetuning Script")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--llm_name', type=str, default=None, help='Name of the local model to finetune')
    parser.add_argument('--lr0', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
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
                bnb_4bit_compute_dtype=torch.bfloat16
            )

    # define model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token_id is None: # autoregressive models' pad token not set by default
        tokenizer.pad_token_id = tokenizer.eos_token_id 

    return model, tokenizer

def get_lora_model(model, device):
    ''' add peft adapter to model '''

    task_type = peft.TaskType.CAUSAL_LM
    
    # prepare for k-bit training
    model = peft.prepare_model_for_kbit_training(model) 
    
    # get peft configs based on architecture (task_type) and fine-tuning method
    config = peft.LoraConfig(task_type=task_type, 
                             inference_mode=False,
                             r=8, 
                             lora_alpha=32,
                             lora_dropout=0.1)

    # wrap model w peft configs
    model = peft.get_peft_model(model, config).to(device)
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
    val_df = pd.read_csv(os.path.join(data_path, 'val.csv'))
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return DatasetDict({"train": train_dataset, "validation": val_dataset})

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
