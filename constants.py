import os

API_KEY = "024d82b3ce26cf982bcca51f3d72905f1df7ba1ca140c4e4c5d838fae7235856"  # Replace with your actual Together.AI API key
API = False
METHOD = "finetuning" # "nshot" or "finetuning"
SHOTS = [0,2,4]
LOCAL_MODEL_DIR = os.path.join(os.getcwd(), "..", "..", "local-models")
LOCAL_MODEL_LIST = ["Llama-3.2-1B", "Llama-3.2-1B-Instruct"]
API_MODEL_LIST = [ "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
                "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            ]
if API:
    LLM_NAME = API_MODEL_LIST[0]
else:
    LLM_NAME = LOCAL_MODEL_LIST[0]

ZS_SYSTEM_PROMPT = """ You are a medical doctor who is expert at writing and analyzing discharge reports. Answer the following questions about the patient in short sentences, you don't need to include all the information:
    - What is age and gender of the patient? and why did they come to the hospital? 
    - What are the relevant pieces of their past medical history? 
    - What are the patient's chief complaint and physical examination findings? 
    - How did their hospital visit go? What did they undergo?
    - What can be said about their discharge instructions? Discharge condition and medications?
    Collect your answers with a summarizing manner in one paragraph. you have to type ||endoftext|| at the end of the paragraph. You are restricted to NOT type anything after ||endoftext||.
    Stop generation after ||endoftext||.
    """

RAW_DATA_FILEPATH = "data/raw/Hospitalization-Summarization.json"
PROCESSED_DATA_DIR = "data/processed"
UNPROCESSED_GENERATED_DIR = "output/unprocessed_generated"
PROCESSED_GENERATED_DIR = "output/processed_generated"


TRAINING_METRICS_SAVE_DIR = "results/training_metrics"
SCORES_SAVE_DIR = "results/score_tables"

ADAPTER_SAVE_DIR = "output/adapters"
SEED = 31
PREPROCESS_SEED = 42
PREPROCESS_TEST_SIZE = 250