import os
import pandas as pd
import json

def load_json(json_file_path):
    # read json file to df 
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    return pd.DataFrame(data)

def build_splits(df):
    dataset_df = pd.DataFrame({'discharge_report': df['instruct'], 'discharge_summary': df['answer']})
    train_df = dataset_df.sample(frac=0.75, random_state=42)
    test_df = dataset_df.drop(train_df.index)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)

def main():
    project_path = os.getcwd()
    raw_data_path = "data/raw/Hospitalization-Summarization.json"
    raw_data_path = os.path.join(project_path, raw_data_path)
    df = load_json(raw_data_path)
    build_splits(df)

if __name__ == '__main__':
    main()