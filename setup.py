import os
from constants import ORIGINAL_DATAPATH, CUSTOM_SPLIT_PATH, TOY_CUSTOM_SPLIT_PATH
from utils import preprocess_data

def main():
    preprocess_data(ORIGINAL_DATAPATH, CUSTOM_SPLIT_PATH)
    
if __name__ == '__main__':
    main()
    
    
