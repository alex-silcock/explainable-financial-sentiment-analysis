import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset, DatasetDict
import argparse
from tqdm import tqdm
import json
from sentiment_analyser import SentimentAnalyser
from db.vector_store import VectorStore

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

    with open("./new/db/train_samples_llm/inferred.json", "r") as f:
        inferred = json.load(f)

    index_store = "./new/db/train-ft-dp/texts.index"
    tweet_store = "./new/db/train-ft-dp/texts.npy"
    explanation_store = "./new/db/train-ft-dp/explanations.npy"
    sentiment_store = "./new/db/train-ft-dp/sentiments.npy"
    vs = VectorStore(index_store=index_store, tweet_store=tweet_store, explanation_store=explanation_store, sentiment_store=sentiment_store)

    print("Length of Vector Store: ", len(vs))
    
    out = {}
    # out of the form:
    # {
    #     "1" : { "prompt" : text }
    # }

    for index, instance in tqdm(enumerate(dataset), desc="Building Prompts", total=len(dataset)):
        prompt = ""

        distance, text, explanation, sentiment, _, _ = vs.find_nearest(instance["text"])
        text = text[0]
        explanation = explanation[0]

        if distance < 1:
            prompt = "Here is an example of a similar text and its explanation: " + text + " (End of information) " + explanation + ". Now try with the following text:"

        if ADD_SENTIMENT:
            prompt += instance["text"] + " (Sentiment: " + sentiment[index] + ") (End of information)"
        else:
            prompt += instance["text"] + " (End of information)"

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