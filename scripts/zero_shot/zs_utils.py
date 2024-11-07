from constants import ZS_SYSTEM_PROMPT, API_KEY, LLM_NAME
from together import Together
from tqdm import tqdm

def single_inference_api(prompt):
    """
    Generates a single summary using the Together.AI API
    Args:
        prompt (str): prompt to generate a summary

    Returns:
        str: generated summary
    """
    client = Together(api_key=API_KEY)
    response = client.completions.create(
        model=LLM_NAME,
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

def generate_summaries(df):
    """
    Generates summaries from df
    Args:
        df (pd.DataFrame): dataframe with discharge reports

    Returns:
        pd.DataFrame: dataframe with generated summaries
    """
    # add empty column for generated summaries  
    generated_summaries = []
    # Generate summaries using the Together.AI API
    for i, trial in tqdm(df.iterrows(), total=df.shape[0], desc="Generating summaries"):
        text = trial["discharge_report"]
        prompt = f"Here is a discharge report of a patient: \n\n{text}\n\n{ZS_SYSTEM_PROMPT}\n\n" 
        generated_summaries.append(single_inference_api(prompt))
    df.loc[:, "generated_summary"] = generated_summaries
    # drop report column
    df = df.drop(columns=["discharge_report"])
    
        
    return df


