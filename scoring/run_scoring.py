# python script to calculate scores from all the runs in the run-outputs folder

import os
import pandas as pd
import json
import numpy as np
import evaluate
import torch
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
from scripts.evaluation.eval_utils import postprocess
from constants import SEED, RUN_OUTPUT_PATH, PROJECT_DIR

from utils import set_seed    

from scoring.bleu import Bleu
from scoring.rouge import Rouge
from scoring.bertscore import BertScore
from scoring.align import AlignScorer
from scoring.UMLSScorer import UMLSScorer

import argparse


def calculate_scores(development_pairs):
    """
    Calculate scores for each metric and return individual scores dataframe and overall score
    Args:
        development_pairs (pd.DataFrame): DataFrame containing gold_summary and gen_summary columns
    Returns:
        pd.DataFrame: DataFrame with individual scores for each metric
        float: Overall score across all metrics
    """
    # Initialize scorers
    bleuScorer = Bleu()
    rougeScorer = Rouge(["rouge1", "rouge2", "rougeL"])
    bertScorer = BertScore()
    alignScorer = AlignScorer()
    medconScorer = UMLSScorer(quickumls_fp=os.path.join(PROJECT_DIR, "quickumls"))
    meteorScorer = evaluate.load("meteor")

    # Get references and hypotheses
    refs = development_pairs["gold_summary"].tolist()
    hyps = development_pairs["gen_summary"].tolist()

    # Calculate scores
    print("Calculating scores...")
    print("bleu scores are being calculated...")
    bleu_scores = bleuScorer(refs=refs, hyps=hyps)
    print("Done!")
    print("rouge scores are being calculated...")
    rouge_scores = rougeScorer(refs=refs, hyps=hyps)
    print("Done!")
    print("bertscore scores are being calculated...")
    bertscore_scores = bertScorer(refs=refs, hyps=hyps)
    print("Done!")
    print("align scores are being calculated...")
    align_scores = alignScorer(refs=refs, hyps=hyps)
    print("Done!")
    print("medcon scores are being calculated...")
    medcon_scores = medconScorer(reference=refs, prediction=hyps)
    print("Done!")
    print("meteor scores are being calculated...")
    meteor_scores = [meteorScorer.compute(references=[r], predictions=[h])["meteor"] for r, h in zip(refs, hyps)]
    print("Done!")

    # Create individual scores dataframe
    scores_df = development_pairs.copy()
    scores_df["rouge1"] = rouge_scores["rouge1"]
    scores_df["rouge2"] = rouge_scores["rouge2"] 
    scores_df["rougeL"] = rouge_scores["rougeL"]
    scores_df["bleu"] = bleu_scores
    scores_df["bertscore"] = bertscore_scores
    scores_df["align"] = align_scores
    scores_df["medcon"] = medcon_scores
    scores_df["meteor"] = meteor_scores

    # Calculate overall score
    metric_averages = {
        "bleu": np.mean(bleu_scores),
        "rouge1": np.mean(rouge_scores["rouge1"]),
        "rouge2": np.mean(rouge_scores["rouge2"]),
        "rougeL": np.mean(rouge_scores["rougeL"]),
        "bertscore": np.mean(bertscore_scores),
        "align": np.mean(align_scores),
        "medcon": np.mean(medcon_scores),
        "meteor": np.mean(meteor_scores)
    }
    overall_score = np.mean(list(metric_averages.values()))

    return scores_df, overall_score

def main():
    # python -m scoring.run_scoring --method qlora
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["qlora", "zeroshot"], required=True) 
    args = parser.parse_args()

    
    print("Starting evaluation process...")
    
    set_seed(SEED)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.method == "qlora":
        table_dir = os.path.join(RUN_OUTPUT_PATH, "qlora")
        runs_dir = os.path.join(RUN_OUTPUT_PATH, "qlora", "runs")
    else:
        table_dir = os.path.join(RUN_OUTPUT_PATH, "zeroshot")
        runs_dir = os.path.join(RUN_OUTPUT_PATH, "zeroshot", "runs")
        
    # check if a table called development_scores.csv exists in the table_dir
    development_scores_path = os.path.join(table_dir, "development_scores.csv")
    if os.path.exists(development_scores_path):
        development_scores_df = pd.read_csv(development_scores_path)
    else:
        development_scores_df = pd.DataFrame(columns=["run_name", "model_name", "gradient_accumulation_steps", "lr0", "lr_scheduler_type", "optimizer_type", "overall_score"])
    
    
    # there are run folders under runs_dir where each folder has all the output files of a run
    for run_folder in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_folder)
         # file path of original-generated pairs that this script will calculate scores for
        postprocessed_outputs_path = os.path.join(run_path, "postprocessed_outputs.csv")
        development_pairs = pd.read_csv(postprocessed_outputs_path)
        individual_scores_df, overall_score = calculate_scores(development_pairs)
        # write individual scores to the table
        individual_scores_df.to_csv(os.path.join(run_path, "individual_scores.csv"), index=False)
        print(f"Individual scores saved to {os.path.join(run_path, 'individual_scores.csv')}")
        run_details = json.load(open(os.path.join(run_path, "run_details.json")))[run_folder]
        # add run's performance to development scores df
        development_scores_df = pd.concat([development_scores_df, pd.DataFrame([{
            "run_name": run_folder,
            "model_name": run_details["llm_name"],
            "gradient_accumulation_steps": run_details["gradient_accumulation_steps"],
            "lr0": "{:.2e}".format(float(run_details["lr0"])), # Format lr0 in scientific notation
            "lr_scheduler_type": run_details["lr_scheduler_type"],
            "optimizer_type": run_details["optimizer_type"],
            "overall_score": overall_score
        }])], ignore_index=True)
        
    # save development scores df to csv, overwrite if it exists. sort by overall_score
    development_scores_df = development_scores_df.sort_values(by="overall_score", ascending=False)
    development_scores_df.to_csv(os.path.join(table_dir, "development_scores.csv"), index=False)
    print(f"Development scores saved to {os.path.join(table_dir, 'development_scores.csv')}")
    

if __name__ == "__main__":
    main()