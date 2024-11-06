import requests
import pandas as pd
import os
import spacy
from together import Together
API_KEY = "22cff3bda0474158a99c200c58b29267fe4540c8af09fbbfb9499e5741e8031d"  # Replace with your actual Together.AI API key



def create_prompt(df):
    few_shot_prompt = "Following is a series of discharge reports and their summaries. See the patterns in the summaries and generate a similar summary for the last report:\n"
    for i, row in df.iterrows():
        few_shot_prompt += f"Report{i+1}:\n {row['discharge_report']}\nSummary{i+1}:\n {row['discharge_summary']}\n -------\n"
    return few_shot_prompt

def generate_summary(text, initial_prompt, model, client):
    """
    Generates a summary using the Together.AI API
    """
    # Define prompt for summarization
    prompt = f"{initial_prompt}Target report:\n {text}\nSummary:\n" 
    print("generating a summary\n")
    response = client.completions.create(
        model=model,
        prompt=prompt,
        min_tokens=500,
        max_tokens=2048,  # Adjust based on your expected summary length
        temperature=0.65,  # Good balance between creativity and consistency
        top_p=0.9,  # Nucleus sampling to maintain coherence
        top_k=40,  # Limit vocabulary while maintaining flexibility
        repetition_penalty=1.2,  # Slight penalty to avoid repetitive text
        stop=["\n -------\n"],
    )
    return response.choices[0].text

# Load and process data
project_path = os.getcwd()
data_path = os.path.join(project_path, "data/processed")
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
test_df = test_df[5:10]
few_shot_df = test_df[:2]
initial_prompt = create_prompt(few_shot_df)
with open(os.path.join(project_path, "output/few_shot_summaries/initial_prompt.txt"), "w") as f:
    f.write(initial_prompt)



    
client = Together(api_key=API_KEY)

model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# Generate summaries using the Together.AI API
test_df["generated_summary"] = test_df["discharge_report"].apply(lambda x: generate_summary(x, initial_prompt, model, client))
# Display results
save_path = os.path.join(project_path, "output/few_shot_summaries/test_generated.csv")
test_df.to_csv(save_path, index=False)


