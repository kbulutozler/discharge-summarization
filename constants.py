import os

API_KEY = "d1671c6fda5a4580817c1800c61c11e04acd00d8dc9142ec95f90f2f8ae46f32"  # Replace with your actual Together.AI API key


ICL_PROMPT = """ 
    You are a medical professional summarizing patient discharge reports. Read the following report carefully and write a concise summary in one paragraph. 
    """



SYSTEM_DIR = "/media/networkdisk/bulut" 
PROJECT_DIR = "/home/bulut/repositories/discharge-summarization" 
ORIGINAL_DATAPATH = os.path.join(PROJECT_DIR, "data/original/Hospitalization-Summarization.json") 
CUSTOM_SPLIT_PATH = os.path.join(PROJECT_DIR, "data/custom_split") # train, val, test
TOY_CUSTOM_SPLIT_PATH = os.path.join(PROJECT_DIR, "data/toy_custom_split") # small batch of train, val, test


LOCAL_MODELS_PATH = os.path.join(SYSTEM_DIR, "local-models") 
RUN_OUTPUT_PATH = os.path.join(SYSTEM_DIR, "run-outputs/discharge-summarization") # trained models saved with loss plot and training logs
CONFIG_PATH = os.path.join(PROJECT_DIR, "config") # config for args and hparams for each run

SEED = 31
PREPROCESS_SEED = 42
PREPROCESS_TEST_SIZE = 250