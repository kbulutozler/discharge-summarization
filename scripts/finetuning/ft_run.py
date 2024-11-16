import pandas as pd
from constants import SEED, ADAPTER_SAVE_DIR, PROCESSED_DATA_DIR, UNPROCESSED_GENERATED_DIR, LOCAL_MODELS_DIR
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
    """
    Main function to run the finetuned model to generate and store the generations. 
    generations get postprocessed in eval_run file under evaluation folder
    """
    args = get_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the LoRA adapter
    project_path = os.getcwd()
    adapter_save_path = os.path.join(project_path, ADAPTER_SAVE_DIR, f'{args.llm_name}')
    base_model_path = os.path.join(LOCAL_MODELS_DIR, args.llm_name)
    processed_data_path = os.path.join(project_path, PROCESSED_DATA_DIR)
    base_model, tokenizer = load_model_and_tokenizer(base_model_path)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_save_path,
        is_trainable=False
    )
    print("model has been loaded from LoRA adapter located at ", adapter_save_path)
    model.eval()
    

    test_df = pd.read_csv(os.path.join(processed_data_path, "test.csv"))
    test_df = test_df[:5]
    test_generated_unprocessed = generate_summaries(args, model, tokenizer, test_df)
    print(test_generated_unprocessed.columns)
    
    save_path = os.path.join(project_path, UNPROCESSED_GENERATED_DIR, "finetune", "test_unprocessed.csv")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    test_generated_unprocessed.to_csv(save_path, index=False)
    print(f"Unprocessed generations saved at {save_path}. run eval_run.py to postprocess and evaluate")
    
    

if __name__ == '__main__':
    main()
