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

    return scores_df, metric_averages, overall_score

def main():
    # python -m scoring.run_scoring --method qlora
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["qlora", "icl"], required=True) 
    parser.add_argument("--test", action='store_true', help="if true, scores will be calculated on model generations of test set")

    args = parser.parse_args()

    
    print("Starting evaluation process...")
    
    set_seed(SEED)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.method == "qlora":
        table_dir = os.path.join(RUN_OUTPUT_PATH, "qlora")
        runs_dir = os.path.join(RUN_OUTPUT_PATH, "qlora", "runs")
    else:
        table_dir = os.path.join(RUN_OUTPUT_PATH, "icl")
        runs_dir = os.path.join(RUN_OUTPUT_PATH, "icl", "runs")
    
        
    if args.test:
        scores_path = os.path.join(table_dir, "test_scores.csv")
    else:
        scores_path = os.path.join(table_dir, "development_scores.csv")
    if os.path.exists(scores_path):
        scores_df = pd.read_csv(scores_path)
    else:
        if args.method == "qlora":
            scores_df = pd.DataFrame(columns=["run_name", "model_name", "gradient_accumulation_steps", "lr0", "lr_scheduler_type", "optimizer_type", "bleu", "rouge1", "rouge2", "rougeL", "bertscore", "align", "medcon", "meteor", "overall_score"])
        else:
            scores_df = pd.DataFrame(columns=["run_name", "bleu", "rouge1", "rouge2", "rougeL", "bertscore", "align", "medcon", "meteor", "overall_score"])
    
    
    # there are run folders under runs_dir where each folder has all the output files of a run
    for run_folder in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_folder)
         # file path of original-generated pairs that this script will calculate scores for
        if args.test:
            postprocessed_outputs_path = os.path.join(run_path, "test_postprocessed_outputs.csv")
        else:
            postprocessed_outputs_path = os.path.join(run_path, "postprocessed_outputs.csv")
        if not os.path.exists(postprocessed_outputs_path):
            print(f"Skipping {run_folder} because file does not exist")
            continue
        pairs = pd.read_csv(postprocessed_outputs_path)
        individual_scores_df, metric_averages, overall_score = calculate_scores(pairs)
        # write individual scores to the table
        if args.test:
            individual_scores_df.to_csv(os.path.join(run_path, "test_individual_scores.csv"), index=False)
            print(f"Individual scores saved to {os.path.join(run_path, 'test_individual_scores.csv')}")
        else:
            individual_scores_df.to_csv(os.path.join(run_path, "individual_scores.csv"), index=False)
            print(f"Individual scores saved to {os.path.join(run_path, 'individual_scores.csv')}")
        
        if args.method == "qlora":
            run_details = json.load(open(os.path.join(run_path, "run_details.json")))[run_folder]
            qlora_metrics = {
                "model_name": run_details["llm_name"],
                "gradient_accumulation_steps": run_details["gradient_accumulation_steps"],
                "lr0": "{:.2e}".format(float(run_details["lr0"])),
                "lr_scheduler_type": run_details["lr_scheduler_type"],
                "optimizer_type": run_details["optimizer_type"]
            }

        # add run's performance to development scores df
        base_metrics = {
            "run_name": run_folder,
            "bleu": metric_averages["bleu"],
            "rouge1": metric_averages["rouge1"],
            "rouge2": metric_averages["rouge2"],
            "rougeL": metric_averages["rougeL"],
            "bertscore": metric_averages["bertscore"],
            "align": metric_averages["align"],
            "medcon": metric_averages["medcon"],
            "meteor": metric_averages["meteor"],
            "overall_score": overall_score
        }
        
        if args.method == "qlora":
            metrics = {**base_metrics, **qlora_metrics}
        else:
            metrics = base_metrics
            
        scores_df = pd.concat([scores_df, pd.DataFrame([metrics])], ignore_index=True)
        
    # save scores df to csv, overwrite if it exists. sort by overall_score
    scores_df = scores_df.sort_values(by="overall_score", ascending=False)
    if args.test:
        scores_df.to_csv(os.path.join(table_dir, "test_scores.csv"), index=False)
    else:
        scores_df.to_csv(os.path.join(table_dir, "development_scores.csv"), index=False)
    print(f"scores saved to {table_dir}")
    

if __name__ == "__main__":
    main()