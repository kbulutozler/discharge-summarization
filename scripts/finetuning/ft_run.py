import pandas as pd
from constants import SEED, FINETUNE_MODEL_NAME, ADAPTER_SAVE_DIR, PROCESSED_DATA_DIR, UNPROCESSED_GENERATED_DIR
from utils import set_seed
from scripts.finetuning.ft_utils import load_model_and_tokenizer, generate_summaries
import torch
import os
from peft import PeftModel


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the LoRA adapter
    project_path = os.getcwd()
    adapter_save_path = os.path.join(project_path, ADAPTER_SAVE_DIR, f'{FINETUNE_MODEL_NAME}')
    base_model_path = os.path.join(project_path, '..', FINETUNE_MODEL_NAME)
    processed_data_path = os.path.join(project_path, PROCESSED_DATA_DIR)
    base_model, tokenizer = load_model_and_tokenizer(base_model_path)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_save_path,
        is_trainable=False
    )
    model.to(device)

    test_df = pd.read_csv(os.path.join(processed_data_path, "test.csv"))
    test_generated_unprocessed = generate_summaries(model, tokenizer, test_df)
    save_path = os.path.join(project_path, UNPROCESSED_GENERATED_DIR, "finetune", "generated.csv")
    test_generated_unprocessed.to_csv(save_path, index=False)
    
    

if __name__ == '__main__':
    main()
