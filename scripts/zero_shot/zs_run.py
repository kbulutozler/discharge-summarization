import pandas as pd
import os
from scripts.zero_shot.zs_utils import generate_summaries
from constants import UNPROCESSED_GENERATED_DIR

def main():
    # Load and process data
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data/processed")
    test_df = pd.read_csv(os.path.join(data_path, "all.csv"))
    test_generated_unprocessed = generate_summaries(test_df)
    
    # Display results
    save_path = os.path.join(project_path, UNPROCESSED_GENERATED_DIR, "zeroshot", "generated.csv")
    test_generated_unprocessed.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()