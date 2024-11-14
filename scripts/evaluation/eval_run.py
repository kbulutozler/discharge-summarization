import os
import pandas as pd
from scripts.evaluation.eval_utils import postprocess, calculate_bertscore, calculate_rouge_l, get_args
from constants import UNPROCESSED_GENERATED_DIR, SCORES_SAVE_DIR, PROCESSED_GENERATED_DIR
import torch
def main():
    """
    Main function to run the evaluation
    set method to benchmark, nshot or finetuned
    args:
        method (str): method to use for evaluation [benchmark, finetune, nshot]
        llm_name (str): name of the model to use for evaluation
    """
    project_path = os.getcwd()
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_generated = pd.read_csv(os.path.join(project_path, UNPROCESSED_GENERATED_DIR, args.method, "test_unprocessed.csv"))
    final_summaries, gold_summaries = postprocess(raw_generated)
    avg_rouge_l, individual_rouge_l = calculate_rouge_l(final_summaries, gold_summaries)
    print("avg rouge l", avg_rouge_l)
    
    avg_bertscore, individual_bertscore = calculate_bertscore(final_summaries, gold_summaries, device)
    print("avg bertscore", avg_bertscore)
    # Create a DataFrame with the required columns
    processed_generated = pd.DataFrame({
        'gold_summary': gold_summaries,
        'gen_summary': final_summaries,
        'rouge_l': [round(score, 2) for score in individual_rouge_l],
        'bertscore_f1': [round(score, 2) for score in individual_bertscore],
    })
    avg_scores = pd.DataFrame({
        'model': args.llm_name,
        'method': args.method,
        'avg_rouge_l': [round(avg_rouge_l, 2)], 
        'avg_bertscore': [round(avg_bertscore, 2)]
    })

    # store average scores in a csv file, update the file if it exists
    score_tables_path = os.path.join(project_path, SCORES_SAVE_DIR)
    if args.method != "benchmark":
        score_tables_path = os.path.join(score_tables_path, "custom_split")
    else:
        score_tables_path = os.path.join(score_tables_path, "benchmark")
    # dont overwrite the file just append unless it is the first time
    if not os.path.exists(os.path.join(score_tables_path, "scores.csv")):
        avg_scores.to_csv(os.path.join(score_tables_path, "scores.csv"), index=False)
    else:
        with open(os.path.join(score_tables_path, "scores.csv"), "a") as f:
            avg_scores.to_csv(f, header=f.tell() == 0, index=False)
    
    # store individual scores for each pair in test set in a csv file: one file per method (finetuning, nshot) and model
    save_path = os.path.join(project_path, PROCESSED_GENERATED_DIR, args.method, args.llm_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    processed_generated.to_csv(os.path.join(save_path, "test_processed_w_scores.csv"), index=False)
    

if __name__ == "__main__":
    main()
    
    
    