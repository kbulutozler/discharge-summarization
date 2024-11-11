API_KEY = "024d82b3ce26cf982bcca51f3d72905f1df7ba1ca140c4e4c5d838fae7235856"  # Replace with your actual Together.AI API key

MODEL_LIST = [  "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
                "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            ]   
LLM_NAME = MODEL_LIST[0]

ZS_SYSTEM_PROMPT = """ You are a medical doctor who is expert at writing and analyzing discharge reports. Answer the following questions about the patient in short sentences, you don't need to include all the information:
    - What is age and gender of the patient? and why did they come to the hospital? 
    - What are the relevant pieces of their past medical history? 
    - What are the patient's chief complaint and physical examination findings? 
    - How did their hospital visit go? What did they undergo?
    - What can be said about their discharge instructions? Discharge condition and medications?
    Collect your answers with a summarizing manner in one paragraph. you have to type ||endoftext|| at the end of the paragraph. You are restricted to NOT type anything after ||endoftext||.
    Stop generation after ||endoftext||.
    """

