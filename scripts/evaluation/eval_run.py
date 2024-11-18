import os
import pandas as pd
from scripts.evaluation.eval_utils import postprocess, calculate_bertscore, calculate_rouge_l, update_json_with_identifier
from constants import SEED, UNPROCESSED_OUTPUT_PATH, PROCESSED_OUTPUT_PATH, RESULT_PATH, RUN_ARGS_PATH, OUTPUT_MODEL_PATH
import torch
from utils import get_args, set_seed
import json



def main():
    """
    Main function to run the evaluation
    set method to benchmark, nshot or finetuned
    args:
        method (str): method to use for evaluation [benchmark, finetune, nshot]
        llm_name (str): name of the model to use for evaluation
    """
    args = get_args("finetune_config")
    set_seed(SEED)
    identifier = args.identifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unprocessed_output = pd.read_csv(os.path.join(UNPROCESSED_OUTPUT_PATH, identifier, "unprocessed_output.csv"))
    final_summaries, gold_summaries = postprocess(unprocessed_output)
    saved_model_path = os.path.join(OUTPUT_MODEL_PATH, f'{args.llm_name}', identifier)

    avg_rouge_l, individual_rouge_l = calculate_rouge_l(final_summaries, gold_summaries)
    
    avg_bertscore, individual_bertscore = calculate_bertscore(final_summaries, gold_summaries, device)
    # Create a DataFrame with the required columns
    processed_output = pd.DataFrame({
        'gold_summary': gold_summaries,
        'gen_summary': final_summaries,
        'rouge_l': [round(score, 3) for score in individual_rouge_l],
        'bertscore_f1': [round(score, 3) for score in individual_bertscore],
    })

    update_json= {'avg_rouge_l': round(avg_rouge_l, 3), 
                'avg_bertscore': round(avg_bertscore, 3)}
    
    # bookkeeping
    update_json_with_identifier(identifier, update_json, os.path.join(RUN_ARGS_PATH, f"{identifier}.json"))
    update_json_with_identifier(identifier, update_json, os.path.join(saved_model_path, "run_args.json"))

    
    # store individual scores for each pair in evaluation set in a csv file: one file per run
    save_path = os.path.join(PROCESSED_OUTPUT_PATH, identifier)
    os.makedirs(save_path, exist_ok=True)
    processed_output.to_csv(os.path.join(save_path, "processed_output.csv"), index=False)
    print(f"processed outputs that have both scores and generated targets saved to {save_path}")
    

if __name__ == "__main__":
    main()
    
    
    