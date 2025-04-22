import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset, DatasetDict
import pandas as pd
import argparse

def main():
    cache_dir = "./.datasets/.stock_market_tweet"
    
    print("Building dataset from scratch.")
    data = pd.read_csv("./new/stock_data.csv")

    data = data.rename(columns={"Text": "text", "Sentiment": "sentiment"})
    # map sentiment of -1 to 0
    data["sentiment"] = data["sentiment"].map({-1: 0, 1: 1})
    

    dataset = Dataset.from_pandas(data)
    dataset = dataset.train_test_split(test_size=0.5)

    print(dataset)

    dataset.save_to_disk(cache_dir)
    print(f"Filtered dataset saved to {cache_dir}")

if __name__ == "__main__":
    main()
    print("Exiting Script.")

