from constants import API_KEY, ICL_PROMPT
from together import Together
from tqdm import tqdm

def build_prompt(examples, new_example, is_basic):
    if is_basic:
        prompt = ""
    else:
        prompt = ICL_PROMPT + "\n\n"
        
    for example in examples:
        prompt += f"**Discharge Report:**\n{example['text']}\n\n**Summary:**\n{example['target']}\n\n"
    prompt += f"**Discharge Report:**\n{new_example['text']}\n\n**Summary:**\n"
    return prompt


def api_inference(model_name, prompt):
    """
    makes a single api call with Together.ai
    Args:
        prompt (str): prompt that includes 

    Returns:
        str: generated summary
    """
    client = Together(api_key=API_KEY)
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        min_tokens=250,
        max_tokens=500,  # Adjust based on your expected summary length
        temperature=0.8,  # Good balance between creativity and consistency
        top_p=0.9,  # Nucleus sampling to maintain coherence
        top_k=40,  # Limit vocabulary while maintaining flexibility
        repetition_penalty=1.2,  # Slight penalty to avoid repetitive text
    )
    return response.choices[0].text

def generate_summaries(model_name, data_df, examples, is_basic):
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
    for i, new_example in tqdm(data_df.iterrows(), total=data_df.shape[0], desc=f"Generating summaries with {model_name} thru api"):
        prompt = build_prompt(examples, new_example, is_basic)
        generated_summaries.append(api_inference(model_name, prompt))
    data_df.loc[:, "generated_summary"] = generated_summaries
    # drop report column
    data_df = data_df.drop(columns=["text"])
    
        
    return data_df

