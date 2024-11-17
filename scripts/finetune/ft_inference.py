import pandas as pd
from constants import SEED, CUSTOM_SPLIT_PATH, UNPROCESSED_OUTPUT_PATH, LOCAL_MODELS_PATH, OUTPUT_MODEL_PATH
from utils import set_seed, get_args
from scripts.finetune.ft_utils import load_model_and_tokenizer, generate_summaries
import torch
import os
from peft import PeftModel

def main():
    """
    Main function to run the finetuned model to generate and store the generations. 
    generations get postprocessed in eval_run file under evaluation folder
    """
    args = get_args("finetune_config")
    set_seed(SEED)
    identifier = args.identifier
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model_path = os.path.join(LOCAL_MODELS_PATH, args.llm_name)
    base_model, tokenizer = load_model_and_tokenizer(base_model_path)
    print("model and tokenizer have been loaded from base model")

    adapter_save_path = os.path.join(OUTPUT_MODEL_PATH, f'{args.llm_name}', identifier)

    model = PeftModel.from_pretrained(
        base_model,
        adapter_save_path,
        is_trainable=False
    )
    print("trained lora adapter loaded.")
    model.eval()
    model.to(device)

    dev_df = pd.read_csv(os.path.join(args.dataset_path, "dev.csv"))
    dev_generated_unprocessed = generate_summaries(args, model, tokenizer, dev_df)
    
    save_path = os.path.join(UNPROCESSED_OUTPUT_PATH, identifier)
    os.makedirs(save_path, exist_ok=True)
        
    dev_generated_unprocessed.to_csv(os.path.join(save_path, "unprocessed_output.csv"), index=False)
    print(f"Unprocessed generations saved at {save_path}. run eval_run.py to postprocess and evaluate")
    
    

if __name__ == '__main__':
    main()
