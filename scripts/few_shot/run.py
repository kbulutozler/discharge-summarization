import requests
import pandas as pd
import os
import spacy
from together import Together
API_KEY = "22cff3bda0474158a99c200c58b29267fe4540c8af09fbbfb9499e5741e8031d"  # Replace with your actual Together.AI API key
def create_prompt(df):
    few_shot_prompt = "Based on the following examples, write a summary for the target report\n\n"
    for _, row in df.iterrows():
        few_shot_prompt += f"<BEGIN_REPORT>\n{row['discharge_report']}\n<END_REPORT>\n<BEGIN_SUMMARY>\n{row['discharge_summary']}\n<END_SUMMARY>\n\n"
    return few_shot_prompt

def generate_summary(text, initial_prompt, model, client):
    """
    Generates a summary using the Together.AI API
    """
    # Define prompt for summarization
    prompt = f"{initial_prompt}\n Now write a summary for the following report:\n<BEGIN_REPORT>\n{text}\n<END_REPORT>\n<BEGIN_SUMMARY>\n" 
    print("generating a summary\n")
    response = client.completions.create(
        model=model,
        prompt=prompt,
        min_tokens=500,
        max_tokens=1000,  # Adjust based on your expected summary length
        temperature=0.8,  # Good balance between creativity and consistency
        top_p=0.9,  # Nucleus sampling to maintain coherence
        top_k=40,  # Limit vocabulary while maintaining flexibility
        repetition_penalty=1.2,  # Slight penalty to avoid repetitive text
    stop=["<END_SUMMARY>", "<BEGIN_REPORT>"],
    )
    return response.choices[0].text

def main():
    # Load and process data
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data/processed")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    #get 10 random rows for small slice
    seed = 42
    small_df = test_df.sample(n=10, random_state=seed)
    few_shot_df = small_df[:3]
    trials_df = small_df[3:6]

    initial_prompt = create_prompt(few_shot_df)
    with open(os.path.join(project_path, "output/few_shot_summaries/initial_prompt.txt"), "w") as f:
        f.write(initial_prompt)



        
    client = Together(api_key=API_KEY)

    model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # add empty column for generated summaries  
    generated_summaries = []
    # Generate summaries using the Together.AI API
    for i, trial in trials_df.iterrows():
        text = trial["discharge_report"]
        generated_summaries.append(generate_summary(text, initial_prompt, model, client))
    trials_df["generated_summary"] = generated_summaries
    # drop report column
    trials_df = trials_df.drop(columns=["discharge_report"])
    # Display results
    save_path = os.path.join(project_path, "output/few_shot_summaries/trials.csv")
    trials_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()