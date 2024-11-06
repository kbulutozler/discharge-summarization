import spacy
import os
import pandas as pd

def write_sent_lines(summaries, file_path):
    for summary in summaries:
        with open(file_path, "a") as f:
            f.write("###SUMMARY: \n")
            for sent in summary:
                f.write(sent + "\n")
            f.write("\n\n")

nlp = spacy.load("en_core_web_sm")
project_path = os.getcwd()
print(project_path)
generated = pd.read_csv(os.path.join(project_path, "output", "few_shot_summaries", "test_generated.csv"))

original_summaries_sentences = []
generated_summaries_sentences = []
for i, row in generated.iterrows():
    original_doc = nlp(row["discharge_summary"])
    original_summaries_sentences.append([sent.text for sent in original_doc.sents])
    generated_doc = nlp(row["generated_summary"])
    generated_summaries_sentences.append([sent.text for sent in generated_doc.sents])
    
    
write_sent_lines(generated_summaries_sentences, os.path.join(project_path, "output/few_shot_summaries/generated_summaries_sentences.txt"))
write_sent_lines(original_summaries_sentences, os.path.join(project_path, "output/few_shot_summaries/original_summaries_sentences.txt"))