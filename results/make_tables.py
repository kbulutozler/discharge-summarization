import os
import json
import pandas as pd

def collect_json_data(base_path):
    data = []
    keys_to_extract = [
        'llm_name', 'method', 'batch_size', 'max_epochs', 'lr0',
        'gradient_accumulation_steps', 'lr_scheduler_type', 'optimizer_type',
        'identifier', 'best_val_loss', 'avg_rouge_l', 'avg_bertscore'
    ]
    
    for folder in os.listdir(base_path):
        run_args_path = os.path.join(base_path, folder, 'run_args')
        if os.path.exists(run_args_path):
            for json_file in os.listdir(run_args_path):
                if json_file.endswith('.json'):
                    file_path = os.path.join(run_args_path, json_file)
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        # Get first (and only) value from the json dict
                        run_data = list(json_data.values())[0]
                        row = {key: run_data.get(key) for key in keys_to_extract}
                        data.append(row)
    
    return pd.DataFrame(data)

base_path = '/xdisk/bethard/kbozler/ds-run-outputs/'
df = collect_json_data(base_path)
df = df.sort_values('avg_rouge_l', ascending=False)
# convert each 0.xxxy to ye-0n
df['lr0'] = df['lr0'].apply(lambda x: '{:.2E}'.format(x))
print(df.head())
df.to_csv('experiments.csv', index=False)
