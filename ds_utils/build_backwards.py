import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset, DatasetDict
import pandas as pd
import argparse
import json

def main():
    cache_dir = "./.datasets/.stock_market_tweet"
    
    print("Building dataset from scratch.")
    data = pd.read_csv("./new/stock_data.csv")

    data = data.rename(columns={"Text": "text", "Sentiment": "sentiment"})
    # map sentiment of -1 to 0
    data["sentiment"] = data["sentiment"].map({-1: 0, 1: 1})

    train = []
    test = []
    other_test = []

    with open("./new/train-mistral7b/prompt_info.json", "r") as f:
        train_prompts = json.load(f)

    for prompt_index in train_prompts:
        sample = train_prompts[prompt_index].get("prompt").split(" (End of information)")[0]
        # find the corresponding tweet in the dataset
        for index, row in data.iterrows():
            if sample == row["text"]:
                train.append(row)

    print("Built train dataset.")

    with open("./new/test-mistral7b/prompt_info.json", "r") as f:
        test_prompts = json.load(f)

    for prompt_index in test_prompts:
        sample = test_prompts[prompt_index].get("prompt").split(" (End of information)")[0]
        # find the corresponding tweet in the dataset
        for index, row in data.iterrows():
            if sample == row["text"]:
                test.append(row)
    
    print("Built test dataset.")

    # append all other samples to other_test
    train_df = pd.DataFrame(train, columns=["text", "sentiment"])
    test_df = pd.DataFrame(test, columns=["text", "sentiment"])

    for _, row in data.iterrows():
        if not ((train_df['text'] == row['text']).any() or (test_df['text'] == row['text']).any()):
            other_test.append(row)

    print("Built other_test dataset.")

    # build dataset with train and test and other_test
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    other_test = pd.DataFrame(other_test)
    # merge test and other_test, but keep test indexed above other_test
    test = pd.concat([test, other_test])

    print("Merged datasets.")

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train),
        "test": Dataset.from_pandas(test),
    })

    print(dataset["train"][0])
    print(dataset["test"][0])

    # dataset = Dataset.from_pandas(data)
    # dataset = dataset.train_test_split(test_size=0.8)

    # print(dataset)

    dataset.save_to_disk(cache_dir)
    print(f"Filtered dataset saved to {cache_dir}")

if __name__ == "__main__":
    main()
    print("Exiting Script.")

