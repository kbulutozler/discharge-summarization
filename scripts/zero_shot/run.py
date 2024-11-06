from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os


def generate_summary(text, model, tokenizer, max_length=512):
    """
    Generates a summary for the input text using the LLaMA model with left padding and a summarization prompt.
    """
    # Define prompt for summarization
    prompt = "Summarize the following discharge report: "
    input_text = prompt + text  # Concatenate prompt with the actual discharge report
    
    # Tokenize input with left padding
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=max_length,
        truncation=False, 
        add_special_tokens=False
    )

    # Move inputs to device and ensure left padding
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary with specified max_length for output
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# Load model and tokenizer
model_path = "/home/kbozler/Documents/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure the EOS token is set as the pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Set model to evaluation mode and disable gradient calculations
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
project_path = os.getcwd()
data_path = os.path.join(project_path, "data/processed")    
test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

# Assume test_df is already loaded with columns "discharge_report" and "discharge_summary"
test_df["generated_summary"] = test_df["discharge_report"].apply(lambda x: generate_summary(x, model, tokenizer))

# Display the first few rows to inspect results
print(test_df[["discharge_report", "discharge_summary", "generated_summary"]].head())