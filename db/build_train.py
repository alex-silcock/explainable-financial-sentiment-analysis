import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset, DatasetDict
import argparse
from tqdm import tqdm
import json
from vector_store import VectorStore

def main():
    # with open("./new/db/train_samples_llm/inferred.json", "r") as f:
    with open("./new/train-mistral7b-finetuned/inferred.json", "r") as f:
        inferred = json.load(f)

    # index_store = "./new/db/train/texts.index"
    # tweet_store = "./new/db/train/texts.npy"
    # explanation_store = "./new/db/train/explanations.npy"
    # sentiment_store = "./new/db/train/sentiments.npy"
    index_store = "./new/db/train-ft-dp/texts.index"
    tweet_store = "./new/db/train-ft-dp/texts.npy"
    explanation_store = "./new/db/train-ft-dp/explanations.npy"
    sentiment_store = "./new/db/train-ft-dp/sentiments.npy"
    vs = VectorStore(index_store=index_store, tweet_store=tweet_store, explanation_store=explanation_store, sentiment_store=sentiment_store, rebuild_if_exists=True)

    for item in inferred:
        tweet = item["text"]
        explanation = item["response"]
        real_sentiment = item["sentiment"]
        predicted_sentiment = item["predicted"]

        # store positive samples
        if real_sentiment == predicted_sentiment:
            res = vs.store_new_text(tweet, explanation, predicted_sentiment, similarity_threshold=SIMILARITY)
        # res = vs.store_new_text(explanation, tweet, predicted_sentiment, similarity_threshold=SIMILARITY)
        
    print(f"Length of vector store is {len(vs)}")
    # print(vs.find_nearest("I am happy."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--similarity', type=float, required=False)
    program_args = parser.parse_args()
    SIMILARITY = program_args.similarity
    
    main()
    print("Exiting Script.")