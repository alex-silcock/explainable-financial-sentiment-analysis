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
    if TRAIN:
        dataset = dataset["train"]
        print("Using prompts for training set.")
    else:
        dataset = dataset["test"]
        print("Using prompts for test set.")

    with open(RESPONSE_FILE, "r") as f:
        responses = json.load(f)

    cleaned = {}
    for index, instance in tqdm(enumerate(responses), desc="Cleaning Responses", total=len(responses)):
        str_index = str(index)

        response = responses[str_index].get("response")

        # if "</think>" in response:
        #     cleaned_response = response.split("</think>")[1]
        if "(End of information) " in response and response.count("(End of information)") > 1:
            cleaned_response = response.split("(End of information)")[2]
        else:
            cleaned_response = response.split("(End of information)")[1]


        if INCLUDE_INPUT:
            cleaned_response = "Input: " + dataset[index]["text"] + " Response: " + cleaned_response

        cleaned[str_index] = {
            "response" : cleaned_response
        }

    if INCLUDE_INPUT:
        output_file = RESPONSE_FILE.split(".json")[0] + "_with_input_cleaned.json"
    else:
        output_file = RESPONSE_FILE.split(".json")[0] + "_cleaned.json"

    with open(output_file, "w+") as f:
        json.dump(cleaned, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--response_file', type=str, required=True)
    parser.add_argument('-i', '--include_input', type=str, required=False)
    parser.add_argument('-t', '--train', type=bool, required=False)

    program_args = parser.parse_args()
    
    RESPONSE_FILE = program_args.response_file
    INCLUDE_INPUT = program_args.include_input
    TRAIN = program_args.train


    main()
    print("Exiting Script.")
