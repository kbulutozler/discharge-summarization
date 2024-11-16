import os
from constants import ORIGINAL_DATAPATH, CUSTOM_SPLIT_PATH, UNPROCESSED_OUTPUT_PATH, PROCESSED_OUTPUT_PATH, OUTPUT_MODEL_PATH, RUN_ARGS_PATH, RESULT_PATH
from utils import preprocess_data

def main():
    if not os.path.exists(CUSTOM_SPLIT_PATH):
        os.makedirs(CUSTOM_SPLIT_PATH)
    if not os.path.exists(UNPROCESSED_OUTPUT_PATH):
        os.makedirs(UNPROCESSED_OUTPUT_PATH)
    if not os.path.exists(PROCESSED_OUTPUT_PATH):
        os.makedirs(PROCESSED_OUTPUT_PATH)
    if not os.path.exists(OUTPUT_MODEL_PATH):
        os.makedirs(OUTPUT_MODEL_PATH)
    if not os.path.exists(RUN_ARGS_PATH):
        os.makedirs(RUN_ARGS_PATH)
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    preprocess_data(ORIGINAL_DATAPATH, CUSTOM_SPLIT_PATH, max_samples=None) # remove max samples for full dataset

if __name__ == '__main__':
    main()
    
    
