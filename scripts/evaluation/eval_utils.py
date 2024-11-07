import spacy
import re
from bert_score import score as bert_score
from rouge_score import rouge_scorer

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

def calculate_bertscore(pred_sequences, gold_sequences, model_type="bert-base-uncased"):
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
    P, R, F1 = bert_score(pred_sequences, gold_sequences, model_type=model_type, lang="en")
    avg_f1 = F1.mean().item()
    return avg_f1, F1.tolist()  # Returning individual F1 scores and average

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
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for gen, gold in zip(pred_sequences, gold_sequences):
        score = scorer.score(gold, gen)
        scores.append(score['rougeL'].fmeasure)

    # Calculate the average ROUGE-L F1 score
    avg_rouge_l = sum(scores) / len(scores) if scores else 0
    return avg_rouge_l, scores  # returning individual scores and average

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
