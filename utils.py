import random
import numpy as np
import torch
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from constants import PREPROCESS_SEED, PREPROCESS_TEST_SIZE, CONFIG_PATH
import yaml
import argparse

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def get_args(config_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config file name')
    parser.add_argument('--identifier', type=str, default=None, help='identifier for the run, usually job id')
    cmd_args = parser.parse_args()
    with open(os.path.join(CONFIG_PATH, f"{cmd_args.config}.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    # Convert dict to namespace
    args = argparse.Namespace(**config[config_type])
    args.identifier = cmd_args.identifier
    return args

def preprocess_data(data_path, save_data_path):
    """ load data from json file to pandas dataframe """
    with open(data_path, 'r') as f:
        data = json.load(f)

    data = pd.DataFrame(data)
    test_size = PREPROCESS_TEST_SIZE

    # remove columns except for instruct and answer
    data = data[['instruct', 'answer']]
    # remove Input:\n from beginning of instruct texts
    data['instruct'] = data['instruct'].apply(lambda x: x.split('Input:\n')[1] if 'Input:\n' in x else x)
    # remove Output:" from the end of instruct texts
    data['instruct'] = data['instruct'].apply(lambda x: x.split('Output:')[0] if 'Output:' in x else x)
    # change column names to text and target
    data.rename(columns={'instruct': 'text', 'answer': 'target'}, inplace=True)
    
    
    # save without splitting, this is useful for zero shot evaluation
    data.to_csv(os.path.join(save_data_path, 'all_data.csv'))
    
    # First obtain test data
    temp_data, test_data = train_test_split(data, test_size=test_size, random_state=PREPROCESS_SEED)
    # Split temp data into train and val 
    train_data, dev_data = train_test_split(temp_data, test_size=0.3, random_state=PREPROCESS_SEED)
    
    # inject ||startoftext|| and ||endoftext|| to train and dev data
    train_data['target'] = train_data['target'].apply(lambda x: "||startoftext|| " + x + " ||endoftext||")
    dev_data['target'] = dev_data['target'].apply(lambda x: "||startoftext|| " + x + " ||endoftext||")
    # save to csv
    train_data.to_csv(os.path.join(save_data_path, 'train.csv'))
    dev_data.to_csv(os.path.join(save_data_path, 'dev.csv'))
    test_data.to_csv(os.path.join(save_data_path, 'test.csv'))
    
    # confirm split sizes
    print(f"Total samples: {len(data)}")
    print(f"Train size: {len(train_data)}")
    print(f"Dev size: {len(dev_data)}")
    print(f"Test size: {len(test_data)}")