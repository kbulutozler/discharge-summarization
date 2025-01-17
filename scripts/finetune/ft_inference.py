import pandas as pd
from constants import SEED, CUSTOM_SPLIT_PATH, LOCAL_MODELS_PATH, RUN_OUTPUT_PATH 
from utils import set_seed, get_args
from scripts.evaluation.eval_utils import postprocess
from scripts.finetune.ft_utils import load_model_and_tokenizer, generate_summaries
import torch
import os
from peft import PeftModel

def main():
    """
    Main function to run the finetuned model to generate and store the generations. 
    generations get postprocessed in eval_run file under evaluation folder
    """
    print("### FINETUNED MODEL INFERENCE ON DEVELOPMENT SET ###")
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

    qlora_run_output_path = os.path.join(RUN_OUTPUT_PATH, 'qlora', 'runs', identifier)
    print(f"Loading trained lora adapter from: {qlora_run_output_path}")

    model = PeftModel.from_pretrained(
        base_model,
        qlora_run_output_path,
        is_trainable=False
    )
    print("Trained lora adapter loaded.")
    model.eval()
    model.to(device)
    print("Model set to evaluation mode and moved to device.")

    dev_df = pd.read_csv(os.path.join(args.dataset_path, "dev.csv"))
    print(f"Development set loaded with {len(dev_df)} samples.")
    inference_outputs_df = generate_summaries(args, model, tokenizer, dev_df)
    
    final_summaries, gold_summaries = postprocess(inference_outputs_df)
    print("Postprocessing completed.")
    # Create a DataFrame with the required columns
    postprocessed_outputs_df = pd.DataFrame({ # gold_summary, gen_summary pairs
        'gold_summary': gold_summaries,
        'gen_summary': final_summaries,
    })
    # store 
    postprocessed_outputs_df.to_csv(os.path.join(qlora_run_output_path, "postprocessed_outputs.csv"), index=False)
    print(f"finetuned model has been used to generate summaries from development set. resulting gold summary generated summary pairs have been saved to {qlora_run_output_path} as postprocessed_outputs.csv")
    
    

if __name__ == '__main__':
    main()
