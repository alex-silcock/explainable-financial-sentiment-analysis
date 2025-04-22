import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset, DatasetDict
import argparse
from tqdm import tqdm
import json


def main():
    # load responses from response dir for any file with batchoutput- in the name
    response_files = [f"{RESPONSE_DIR}/{f}" for f in os.listdir(RESPONSE_DIR) if f.startswith("batchoutput")]

    responses_cleaned = []

    for response_file in response_files:
        with open(response_file, "r") as f:
            responses = [json.loads(line) for line in f]
            
        for i in range(len(responses)):
            response = responses[i].get("response").get("body").get("choices")[0].get("message").get("content")
            responses_cleaned.append(response)
            
    cleaned = {}

    for index, instance in tqdm(enumerate(responses_cleaned), desc="Cleaning Responses", total=len(responses_cleaned)):
        cleaned[index] = {
            "response" : instance
        }

    output_file = RESPONSE_DIR + "/responses_cleaned.json"
    with open(output_file, "w+") as f:
        json.dump(cleaned, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--response_dir', type=str, required=True)
    program_args = parser.parse_args()
    
    RESPONSE_DIR = program_args.response_dir

    main()
    print("Exiting Script.")
