import pandas as pd
import os
from scripts.evaluation.eval_utils import postprocess
from constants import LOCAL_MODELS_PATH, RUN_OUTPUT_PATH
from scripts.finetune.ft_utils import load_model_and_tokenizer, generate_summaries
from peft import PeftModel
import torch
import json


import argparse
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", type=str, required=True) 
    parser.add_argument("--dataset_path", type=str, required=True) # path that has train, dev, test csv files

    args = parser.parse_args()
    qlora_run_output_path = os.path.join(RUN_OUTPUT_PATH, 'qlora', 'saved-runs', args.identifier)
    qlora_test_output_path = os.path.join(RUN_OUTPUT_PATH, 'qlora', 'runs', args.identifier)
    os.makedirs(qlora_test_output_path, exist_ok=True)
    run_details = json.load(open(os.path.join(qlora_run_output_path, "run_details.json")))
    # write run_details to test output path so it can be used during run_scoring
    json.dump(run_details, open(os.path.join(qlora_test_output_path, "run_details.json"), "w"))
    run_details = run_details[args.identifier]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model_path = os.path.join(LOCAL_MODELS_PATH, run_details["llm_name"])
    print(f"Loading model from path: {base_model_path}")
    base_model, tokenizer = load_model_and_tokenizer(base_model_path)
    print("Model and tokenizer have been loaded from base model.")

    model = PeftModel.from_pretrained(
        base_model,
        qlora_run_output_path,
        is_trainable=False
    )
    print(f"Trained lora adapter loaded from: {qlora_run_output_path}.")
    model.eval()
    model.to(device)
    print("Model set to evaluation mode and moved to device.")
    
    
    data = pd.read_csv(os.path.join(args.dataset_path, "test.csv"))
    print(f"test dataset has been loaded with {len(data)} samples.")

        
    inference_outputs_df = generate_summaries(run_details, model, tokenizer, data)
    inference_outputs_df.to_csv(os.path.join(qlora_test_output_path, "test_unprocessed_outputs.csv"), index=False)
    print(f"unprocessed inference outputs have been saved to {qlora_test_output_path} as test_unprocessed_outputs.csv")
    
    final_summaries, gold_summaries = postprocess(inference_outputs_df)
    print("Postprocessing completed.")
    # Create a DataFrame with the required columns
    postprocessed_outputs_df = pd.DataFrame({ # gold_summary, gen_summary pairs
        'gold_summary': gold_summaries,
        'gen_summary': final_summaries,
    })
    # store 
    postprocessed_outputs_df.to_csv(os.path.join(qlora_test_output_path, "test_postprocessed_outputs.csv"), index=False)
    print(f"finetuned model has been used to generate summaries from given test set. resulting gold summary generated summary pairs have been saved to {qlora_test_output_path} as test_postprocessed_outputs.csv")
    
    


if __name__ == "__main__":
    main()