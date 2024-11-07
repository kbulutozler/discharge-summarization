import spacy
import os
import pandas as pd
import re

def main():
    nlp = spacy.load("en_core_web_sm")
    project_path = os.getcwd()
    generated = pd.read_csv(os.path.join(project_path, "output", "zero_shot_summaries", "qa_style_trials.csv"))
    generated_summaries = []
    for i, row in generated.iterrows():
        generated_doc = nlp(row["generated_summary"].replace('\n', ' ').replace('\r', ' '))
        list_of_sentences = [sent.text for sent in generated_doc.sents if sent.text.strip()]
        final_summary = " ".join(list_of_sentences)
        generated_summaries.append(final_summary)
    
    pattern = r'(<[^>]+>)|(\|\|.+?\|\|)|(\n{2,})|(\s+\)|\(\s+)|(endoftext)'

    final_summaries = []
    for summary in generated_summaries:
        sliced_text = re.split(pattern, summary, maxsplit=1)[0]
        final_summaries.append(sliced_text)
    
    for summary in final_summaries:
        print(summary)
        print("\n\n")

if __name__ == "__main__":
    main()
    
    
    