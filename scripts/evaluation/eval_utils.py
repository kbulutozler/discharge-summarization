import spacy
import re
import numpy as np
import argparse
import json

import evaluate








def update_json_with_identifier(identifier, update_json, save_path):
    with open(save_path, 'r') as f:
        run_args = json.load(f)
    for key, value in update_json.items():
        run_args[identifier][key] = value
    with open(save_path, 'w') as f:
        json.dump(run_args, f, indent=4)

def postprocess(df):
    """
    Postprocesses the generated summaries
    Args:
        df (pd.DataFrame): dataframe with generated summaries

    Returns:
        list: list of cleaned summaries
        list: list of gold summaries
    """
    generated_summaries = split_and_merge(df)
    generated_summaries = clean_stop_tokens(generated_summaries)
    gold_summaries = df["target"].tolist()
    for i, summary in enumerate(gold_summaries):
        gold_summaries[i] = summary.replace("||startoftext||", "").replace("||endoftext||", "") # remove to only show the summary

    return generated_summaries, gold_summaries


def split_and_merge(df):
    """
    Splits the generated summaries into sentences and merges them into a single string
    Args:
        df (pd.DataFrame): dataframe with generated summaries

    Returns:
        list: list of merged summaries
    """
    nlp = spacy.load("en_core_web_sm") # python -m spacy download en_core_web_sm
    generated_summaries = []
    for i, row in df.iterrows():
        generated_doc = nlp(row["generated_summary"].replace('\n', ' ').replace('\r', ' '))
        list_of_sentences = [sent.text for sent in generated_doc.sents if sent.text.strip()]
        summary = " ".join(list_of_sentences) # merge sentences into a single string
        generated_summaries.append(summary)
    return generated_summaries

def clean_stop_tokens(sequences): # not 100% success rate
    """
    Extracts text between startoftext and endoftext
    Args:
        sequences (list): list of sequences to clean

    Returns:
        list: list of cleaned sequences
    """
    cleaned_sequences = []
    for sequence in sequences:
        pattern = r'startoftext(.*?)endoftext'
        match = re.search(pattern, sequence, re.IGNORECASE)
        if match:
            cleaned_text = match.group(1).strip()
        else:
            cleaned_text = sequence.strip()
        cleaned_sequences.append(cleaned_text)
    return cleaned_sequences

def calculate_bertscore(pred_sequences, gold_sequences, device):
    """
    Calculates BERTScore Precision, Recall, and F1 for the generated summaries
    Args:
        pred_sequences (list): list of generated summaries
        gold_sequences (list): list of gold summaries
        model_type (str, optional): model type to use for BERTScore. Defaults to "bert-base-uncased".

    Returns:
        float: average BERTScore F1 score
        list: list of individual BERTScore F1 scores
    """
    # Calculate BERTScore Precision, Recall, and F1
    bertscore = evaluate.load("bertscore")
    F1 = bertscore.compute(predictions=pred_sequences, references=gold_sequences, lang="en", device=device)["f1"]
    avg_f1 = np.mean(F1)
    return avg_f1, F1  # Returning individual F1 scores and average

def calculate_rouge_l(pred_sequences, gold_sequences):
    """
    Calculates ROUGE-L score for the generated summaries
    Args:
        pred_sequences (list): list of generated summaries
        gold_sequences (list): list of gold summaries

    Returns:
        float: average ROUGE-L score
        list: list of individual ROUGE-L scores
    """
    rouge = evaluate.load("rouge")
    
    scores = rouge.compute(predictions=pred_sequences, references=gold_sequences, use_aggregator=False)

    avg_rouge_l = np.mean(scores['rougeL'])
    return avg_rouge_l, scores['rougeL']  # returning individual scores and average

