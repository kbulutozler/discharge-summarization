import pandas as pd
import os
from scripts.n_shot.n_shot_utils import generate_summaries
from scripts.evaluation.eval_utils import postprocess

import argparse
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True) 
    parser.add_argument("--dataset_path", type=str, required=True) # path that has train, dev, test csv files
    parser.add_argument("--is_basic", action='store_true', help="if true, use basic prompting, otherwise use informative prompting")
    parser.add_argument("--shot", type=int, required=True) # 0 if zeroshot, n if n-shot
    parser.add_argument("--output_path", type=str, required=True) # path to save outputs
    parser.add_argument("--test", action='store_true', help="if true, generations will be for test set, otherwise dev set")

    args = parser.parse_args()
    print(args)
    if args.is_basic:
        run_folder_name = f"basic_prompt_{args.shot}_shot_{args.model.split('/')[-1]}"
    else:
        run_folder_name = f"informative_prompt_{args.shot}_shot_{args.model.split('/')[-1]}"
    nshot_output_path = os.path.join(args.output_path, 'icl', 'runs', run_folder_name)
    os.makedirs(nshot_output_path, exist_ok=True)
    print(f"n-shot method outputs will be saved to: {nshot_output_path}")
    
    examples = []
    if args.shot > 0:
        train_df = pd.read_csv(os.path.join(args.dataset_path, "train.csv"))
        # get random n examples
        selected_examples = train_df.sample(n=args.shot)
        for i,row in selected_examples.iterrows():
            examples.append({
                "text": row["text"],
                "target": row["target"].replace("||startoftext||", "").replace("||endoftext||", "")
            })
    if args.test:
        data = pd.read_csv(os.path.join(args.dataset_path, "test.csv"))
        print(f"test dataset has been loaded with {len(data)} samples.")
    else:
        data = pd.read_csv(os.path.join(args.dataset_path, "dev.csv"))
        print(f"dev dataset has been loaded with {len(data)} samples.")

    
    inference_outputs_df = generate_summaries(args.model, data, examples, args.is_basic)
    if args.test:
        inference_outputs_df.to_csv(os.path.join(nshot_output_path, "test_unprocessed_outputs.csv"), index=False)
    else:
        inference_outputs_df.to_csv(os.path.join(nshot_output_path, "unprocessed_outputs.csv"), index=False)
    print(f"unprocessed inference outputs have been saved to {nshot_output_path} as unprocessed_outputs.csv")
    
    final_summaries, gold_summaries = postprocess(inference_outputs_df)
    print("Postprocessing completed.")
    # Create a DataFrame with the required columns
    postprocessed_outputs_df = pd.DataFrame({ # gold_summary, gen_summary pairs
        'gold_summary': gold_summaries,
        'gen_summary': final_summaries,
    })
    # store 
    if args.test:
        postprocessed_outputs_df.to_csv(os.path.join(nshot_output_path, "test_postprocessed_outputs.csv"), index=False)
    else:
        postprocessed_outputs_df.to_csv(os.path.join(nshot_output_path, "postprocessed_outputs.csv"), index=False)
    print(f"n-shot model has been used to generate summaries from given dataset. resulting gold summary generated summary pairs have been saved to {nshot_output_path}")
    
    


if __name__ == "__main__":
    main()