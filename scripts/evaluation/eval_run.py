import os
import pandas as pd
from scripts.evaluation.eval_utils import postprocess, calculate_bertscore, calculate_rouge_l
from constants import SEED, UNPROCESSED_OUTPUT_PATH, PROCESSED_OUTPUT_PATH, RESULT_PATH
import torch
from utils import get_args, set_seed

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
    # read identifier from file
    if os.path.exists("identifier.txt"):
        with open("identifier.txt", "r") as f: 
            identifier = f.read() 
    else:
        raise ValueError("identifier.txt not found. Complete a run first...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_generated = pd.read_csv(os.path.join(UNPROCESSED_OUTPUT_PATH, identifier, "unprocessed_output.csv"))
    final_summaries, gold_summaries = postprocess(raw_generated)
    final_summaries = final_summaries[:10]
    gold_summaries = gold_summaries[:10]
    avg_rouge_l, individual_rouge_l = calculate_rouge_l(final_summaries, gold_summaries)
    print("avg rouge l", avg_rouge_l)
    
    avg_bertscore, individual_bertscore = calculate_bertscore(final_summaries, gold_summaries, device)
    print("avg bertscore", avg_bertscore)
    # Create a DataFrame with the required columns
    processed_generated = pd.DataFrame({
        'gold_summary': gold_summaries,
        'gen_summary': final_summaries,
        'rouge_l': [round(score, 3) for score in individual_rouge_l],
        'bertscore_f1': [round(score, 3) for score in individual_bertscore],
    })
    avg_scores = {
        'model': args.llm_name,
        'method': args.method,
        'identifier': identifier,
        'avg_rouge_l': round(avg_rouge_l, 3), 
        'avg_bertscore': round(avg_bertscore, 3)
    }
    avg_scores_df = pd.DataFrame([avg_scores])  # Convert dict to DataFrame
    # store average scores in a csv file, update the file if it exists
    if args.method != "benchmark":
        score_table_path = os.path.join(RESULT_PATH, "custom_split")
    else:
        score_table_path = os.path.join(RESULT_PATH, "benchmark")
    if not os.path.exists(score_table_path):   
        os.makedirs(score_table_path)

    performance_scores_path = os.path.join(score_table_path, "performance_scores.csv")
    if os.path.exists(performance_scores_path):
        avg_scores_df.to_csv(performance_scores_path, mode='a', header=False, index=False)
    else:
        avg_scores_df.to_csv(performance_scores_path, index=False)

    
    # store individual scores for each pair in test set in a csv file: one file per run
    save_path = os.path.join(PROCESSED_OUTPUT_PATH, identifier)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    processed_generated.to_csv(os.path.join(save_path, "processed_output.csv"), index=False)
    

if __name__ == "__main__":
    main()
    
    
    