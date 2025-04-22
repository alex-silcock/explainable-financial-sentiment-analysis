import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset, DatasetDict
import argparse
from tqdm import tqdm
import json

def main():
    cache_dir = "./.datasets/.stock_market_tweet"
    dataset = DatasetDict.load_from_disk(cache_dir)
    train_dataset = dataset["train"]

    with open(RESPONSE_FILE, "r") as f:
        responses = json.load(f)

    min_index = min(len(train_dataset), len(responses)) - 1

    out = []

    for index, instance in tqdm(enumerate(train_dataset), desc="Building Fine Tuning DS", total=len(train_dataset)):
        if INCLUDE_INPUT:
            text = "Input: " + instance["text"] + " Response: " + responses[str(index)].get("response")
        else:
            text = responses[str(index)].get("response")

        out.append({
            "text" : text,
            "label" : instance["sentiment"]
        })

        if index == min_index:
            break
    

    # dataset_out = "./.datasets/.stock_market_tweet_for_finetune_4omini"
    dataset_out = OUTPUT_DATASET
    dataset = Dataset.from_list(out)
    dataset.save_to_disk(dataset_out)
    print(f"Saved dataset to {dataset_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--response_file', type=str, required=True)
    parser.add_argument('-o', '--output_dataset', type=str, required=True)
    parser.add_argument('-i', '--include_input', type=str, required=False)
    program_args = parser.parse_args()
    
    RESPONSE_FILE = program_args.response_file
    OUTPUT_DATASET = program_args.output_dataset
    INCLUDE_INPUT = program_args.include_input
    
    main()
    print("Exiting Script.")