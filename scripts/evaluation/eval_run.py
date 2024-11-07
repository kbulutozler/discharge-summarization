import os
import pandas as pd
from scripts.evaluation.eval_utils import postprocess, calculate_bertscore, calculate_rouge_l
from constants import LLM_NAME

def main():
    """
    Main function to run the evaluation
    set method to zeroshot, n-shot or finetuned
    """
    method = "zeroshot"
    project_path = os.getcwd()
    raw_generated = pd.read_csv(os.path.join(project_path, "output", "zs_summaries", "test_zs_unprocessed.csv"))
    final_summaries, gold_summaries = postprocess(raw_generated)
    avg_rouge_l, individual_rouge_l = calculate_rouge_l(final_summaries, gold_summaries)
    avg_bertscore, individual_bertscore = calculate_bertscore(final_summaries, gold_summaries)

    # Create a DataFrame with the required columns
    individual_scores = pd.DataFrame({
        'gold_summary': gold_summaries,
        'gen_summary': final_summaries,
        'rouge_l': [round(score, 2) for score in individual_rouge_l],
        'bertscore_f1': [round(score, 2) for score in individual_bertscore]
    })
    avg_scores = pd.DataFrame({
        'model': LLM_NAME,
        'method': method,
        'avg_rouge_l': [round(avg_rouge_l, 2)], 
        'avg_bertscore': [round(avg_bertscore, 2)]  
    })

    # fill out the big table
    score_tables_path = os.path.join(project_path, "results", "score_tables")
    # dont overwrite the file just append
    with open(os.path.join(score_tables_path, "all.csv"), "a") as f:
        avg_scores.to_csv(f, header=f.tell() == 0, index=False)
    
    # fill out the method-model specific scores table for all test samples
    score_tables_path = os.path.join(score_tables_path, method, LLM_NAME)
    if not os.path.exists(score_tables_path):
        os.makedirs(score_tables_path)
    
    individual_scores.to_csv(os.path.join(score_tables_path, "scores_w_outputs.csv"), index=False)
    

if __name__ == "__main__":
    main()
    
    
    