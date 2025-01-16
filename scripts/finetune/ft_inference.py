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
    print("Starting main function.")
    args = get_args("finetune_config")
    print(f"Arguments loaded: {args}")
    set_seed(SEED)
    identifier = args.identifier
    print(f"Seed set to: {SEED}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model_path = os.path.join(LOCAL_MODELS_PATH, args.llm_name)
    print(f"Loading model from path: {base_model_path}")
    base_model, tokenizer = load_model_and_tokenizer(base_model_path)
    print("Model and tokenizer have been loaded from base model.")

    adapter_save_path = os.path.join(OUTPUT_MODEL_PATH, f'{args.llm_name}', identifier)
    print(f"Loading trained lora adapter from: {adapter_save_path}")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_save_path,
        is_trainable=False
    )
    print("Trained lora adapter loaded.")
    model.eval()
    model.to(device)
    print("Model set to evaluation mode and moved to device.")

    dev_df = pd.read_csv(os.path.join(args.dataset_path, "dev.csv"))
    print(f"Development dataset loaded with {len(dev_df)} samples.")
    dev_generated_unprocessed = generate_summaries(args, model, tokenizer, dev_df)
    print("Generated unprocessed summaries.")

    save_path = os.path.join(UNPROCESSED_OUTPUT_PATH, identifier)
    os.makedirs(save_path, exist_ok=True)
    print(f"Output directory created at: {save_path}")
        
    dev_generated_unprocessed.to_csv(os.path.join(save_path, "unprocessed_output.csv"), index=False)
    print(f"Unprocessed generations saved at {save_path}. Run eval_run.py to postprocess and evaluate.")
    
    

if __name__ == '__main__':
    main()
