import requests
import pandas as pd
import os
import spacy
import numpy as np
from together import Together
from tqdm import tqdm  # Import tqdm
API_KEY = "024d82b3ce26cf982bcca51f3d72905f1df7ba1ca140c4e4c5d838fae7235856"  # Replace with your actual Together.AI API key


def create_prompt():
    return """ You are a medical doctor who is expert at writing and analyzing discharge reports. Answer the following questions about the patient in short sentences, you don't need to include all the information:
    - What is age and gender of the patient? and why did they come to the hospital? 
    - What are the relevant pieces of their past medical history? 
    - What are the patient's chief complaint and physical examination findings? 
    - How did their hospital visit go? What did they undergo?
    - What can be said about their discharge instructions? Discharge condition and medications?
    Collect your answers with a summarizing manner in one paragraph. you have to type ||endoftext|| at the end of the paragraph. You are restricted to NOT type anything after ||endoftext||.
    Stop generation after ||endoftext||.
    """
    
def single_inference(prompt, model, client):
    response = client.completions.create(
        model=model,
        prompt=prompt,
        min_tokens=250,
        max_tokens=500,  # Adjust based on your expected summary length
        temperature=0.8,  # Good balance between creativity and consistency
        top_p=0.9,  # Nucleus sampling to maintain coherence
        top_k=40,  # Limit vocabulary while maintaining flexibility
        repetition_penalty=1.2,  # Slight penalty to avoid repetitive text
        stop=["||endoftext||"],
    )
    return response.choices[0].text

def generate_summary(text, system_prompt, model, client):
    """
    Generates a summary using the Together.AI API
    """
    # Define prompt for summarization
    prompt = f"Here is a discharge report of a patient: \n\n{text}\n\n{system_prompt}\n\n" 
        
    return single_inference(prompt, model, client)

def main():
    # Load and process data
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data/processed")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    #get 10 random rows for small slice
    seed = 11
    trials_df = test_df.sample(n=20, random_state=seed)


    system_prompt = create_prompt()
    with open(os.path.join(project_path, "output/zero_shot_summaries/qa_style_system_prompt.txt"), "w") as f:
        f.write(system_prompt)



        
    client = Together(api_key=API_KEY)

    model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    # add empty column for generated summaries  
    generated_summaries = []
    # Generate summaries using the Together.AI API
    for i, trial in tqdm(trials_df.iterrows(), total=trials_df.shape[0], desc="Generating summaries"):
        text = trial["discharge_report"]
        generated_summaries.append(generate_summary(text, system_prompt, model, client))
    trials_df.loc[:, "generated_summary"] = generated_summaries
    # drop report column
    trials_df = trials_df.drop(columns=["discharge_report"])
    # Display results
    save_path = os.path.join(project_path, "output/zero_shot_summaries/qa_style_trials.csv")
    trials_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()