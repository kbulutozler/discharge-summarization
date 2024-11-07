import spacy
import os
import pandas as pd
import re

def split_and_merge(df):
    nlp = spacy.load("en_core_web_sm")
    generated_summaries = []
    for i, row in df.iterrows():
        generated_doc = nlp(row["generated_summary"].replace('\n', ' ').replace('\r', ' '))
        list_of_sentences = [sent.text for sent in generated_doc.sents if sent.text.strip()]
        final_summary = " ".join(list_of_sentences)
        generated_summaries.append(final_summary)
    return generated_summaries

def clean_stop_tokens(sequences): # slice until first stop token
    pattern = r'(<[^>]+>)|(\|\|.+?\|\|)|(\n{2,})|(\s+\)|\(\s+)|(endoftext)'
    cleaned_sequences = []
    for sequence in sequences:
        sliced_text = re.split(pattern, sequence, maxsplit=1)[0]
        cleaned_sequences.append(sliced_text)
    return cleaned_sequences
def main():
    project_path = os.getcwd()
    generated = pd.read_csv(os.path.join(project_path, "output", "zero_shot_summaries", "qa_style_trials.csv"))
    generated_summaries = split_and_merge(generated)
    final_summaries = clean_stop_tokens(generated_summaries)


if __name__ == "__main__":
    main()
    
    
    