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

    train_dataset = train_dataset.rename_column("sentiment", "label")
    
    dataset_out = "./.datasets/.stock_market_tweet_for_finetune_baseline"
    dataset = train_dataset
    dataset.save_to_disk(dataset_out)

if __name__ == "__main__":
    main()
    print("Exiting Script.")