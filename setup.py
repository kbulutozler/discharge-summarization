import os
from constants import RAW_DATA_FILEPATH, PROCESSED_DATA_DIR
from utils import preprocess_data

def main():
    project_path = os.getcwd()
    data_path = os.path.join(project_path, RAW_DATA_FILEPATH)
    save_data_path = os.path.join(project_path, PROCESSED_DATA_DIR)
    preprocess_data(data_path, save_data_path, max_samples=40) # remove max samples for full dataset

if __name__ == '__main__':
    main()