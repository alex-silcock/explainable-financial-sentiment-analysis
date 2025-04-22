import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset, DatasetDict
import argparse
from tqdm import tqdm
import json
from sentiment_analyser import SentimentAnalyser

def main():
    cache_dir = "./.datasets/.stock_market_tweet"
    dataset = DatasetDict.load_from_disk(cache_dir)

    if TRAIN:
        dataset = dataset["train"]
        print("Building prompts for training set.")
        
    else:
        dataset = dataset["test"]
        print("Building prompts for test set.")

    if NUM_SAMPLES:
        dataset = dataset.select(range(NUM_SAMPLES))

    if ADD_SENTIMENT:
        sentiment_analyser = SentimentAnalyser(model_path="./models/bert-large-uncased-finetuned-twitter-sentiment")
        sentiment = sentiment_analyser.infer_texts_batch(dataset["text"], batch_size=16)


    out = {}
    # out of the form:
    # {
    #     "1" : { "prompt" : text }
    # }

    for index, instance in tqdm(enumerate(dataset), desc="Building Prompts", total=len(dataset)):
        if ADD_SENTIMENT:
            prompt = instance["text"] + " (Sentiment: " + sentiment[index] + ") (End of information)"
        else:
            prompt = instance["text"] + " (End of information)"

        out[index] = {
            "prompt" : prompt
        }

    with open(f"{OUTPUT_DIR}/prompt_info.json", "w+") as f:
        json.dump(out, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-n', '--num_samples', type=int, required=False)
    parser.add_argument('-t', '--train', type=bool, required=False)
    parser.add_argument('-s', '--add_sentiment', type=bool, required=False)

    program_args = parser.parse_args()
    
    OUTPUT_DIR = program_args.output_dir
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    NUM_SAMPLES = program_args.num_samples
    TRAIN = program_args.train
    ADD_SENTIMENT = program_args.add_sentiment

    main()
    print("Exiting Script.")
