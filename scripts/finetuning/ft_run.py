import pandas as pd
from constants import SEED, ADAPTER_SAVE_DIR, PROCESSED_DATA_DIR, UNPROCESSED_GENERATED_DIR
from utils import set_seed
from scripts.finetuning.ft_utils import load_model_and_tokenizer, generate_summaries
import torch
import os
import argparse
from peft import PeftModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the LoRA adapter
    project_path = os.getcwd()
    adapter_save_path = os.path.join(project_path, ADAPTER_SAVE_DIR, f'{args.llm_name}')
    base_model_path = os.path.join(project_path, '..', '..', "local-models", args.llm_name)
    processed_data_path = os.path.join(project_path, PROCESSED_DATA_DIR)
    base_model, tokenizer = load_model_and_tokenizer(base_model_path)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_save_path,
        is_trainable=False
    )
    model.to(device)
    model.eval()

    test_df = pd.read_csv(os.path.join(processed_data_path, "test.csv"))
    test_generated_unprocessed = generate_summaries(model, tokenizer, test_df)
    save_path = os.path.join(project_path, UNPROCESSED_GENERATED_DIR, "finetune", "generated.csv")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    test_generated_unprocessed.to_csv(save_path, index=False)
    
    

if __name__ == '__main__':
    main()
