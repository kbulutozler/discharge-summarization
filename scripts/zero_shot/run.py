import requests
import pandas as pd
import os

from together import Together
API_KEY = "22cff3bda0474158a99c200c58b29267fe4540c8af09fbbfb9499e5741e8031d"  # Replace with your actual Together.AI API key




def generate_summary(text, model, client):
    """
    Generates a summary using the Together.AI API
    """
    # Define prompt for summarization
    prompt = "Summarize the following discharge report: "
    input_text = prompt + text

    response = client.completions.create(
        model=model,
        prompt=input_text,
        max_tokens=4096,
    )
    return response.choices[0].text

# Load and process data
project_path = os.getcwd()
data_path = os.path.join(project_path, "data/processed")
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
test_df = test_df[:10]

client = Together(api_key=API_KEY)

model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# Generate summaries using the Together.AI API
test_df["generated_summary"] = test_df["discharge_report"].apply(lambda x: generate_summary(x, model, client))

# Display results
save_path = os.path.join(project_path, "output/zero_shot_summaries/test_generated.csv")
test_df.to_csv(save_path, index=False)
