import argparse
from datasets import Dataset, DatasetDict
import json

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    print("Loading dataset from cache.")
    try:
        dataset = Dataset.load_from_disk(load_dataset)
    except:
        dataset = DatasetDict.load_from_disk(load_dataset)
    print(dataset)
    try:
        print(dataset[0])
        print(dataset[1])
        print(dataset[2])
    except:
        for index, value in enumerate(dataset["train"]):
            if value.get("contains_trump_tweet"):
                print(json.dumps(value, indent=4))

            if index > 700:
                exit()

        print(dataset["train"][0])
        print(dataset["train"][1])
        print(dataset["train"][2])
        print(dataset["train"][3])
        print(dataset["train"][4])
        print(dataset["train"][5])
        print(dataset["train"][6])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    program_args = parser.parse_args()
    load_dataset = program_args.dataset

    main()
    print("Exiting Script.")