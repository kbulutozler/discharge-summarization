import spacy
import re
import evaluate
import numpy as np


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
    post_summaries = clean_stop_tokens(generated_summaries)
    gold_summaries = df["discharge_summary"].tolist()
    return post_summaries, gold_summaries

def calculate_bertscore(pred_sequences, gold_sequences):
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
    F1 = bertscore.compute(predictions=pred_sequences, references=gold_sequences, lang="en")["f1"]
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

def split_and_merge(df):
    """
    Splits the generated summaries into sentences and merges them into a single string
    Args:
        df (pd.DataFrame): dataframe with generated summaries

    Returns:
        list: list of merged summaries
    """
    nlp = spacy.load("en_core_web_sm")
    generated_summaries = []
    for i, row in df.iterrows():
        generated_doc = nlp(row["generated_summary"].replace('\n', ' ').replace('\r', ' '))
        list_of_sentences = [sent.text for sent in generated_doc.sents if sent.text.strip()]
        final_summary = " ".join(list_of_sentences)
        generated_summaries.append(final_summary)
    return generated_summaries

def clean_stop_tokens(sequences): # slice until first stop token
    """
    Cleans stop tokens from the sequences
    Args:
        sequences (list): list of sequences to clean

    Returns:
        list: list of cleaned sequences
    """
    pattern = r'(<[^>]+>)|(\|\|.+?\|\|)|(\n{2,})|(\s+\)|\(\s+)|(endoftext)'
    cleaned_sequences = []
    for sequence in sequences:
        sliced_text = re.split(pattern, sequence, maxsplit=1)[0]
        cleaned_sequences.append(sliced_text)
    return cleaned_sequences
