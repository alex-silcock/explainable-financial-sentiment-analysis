import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset, DatasetDict
import argparse
from tqdm import tqdm
import json
from vector_store import VectorStore
# from summariser import Summariser

def main():
    # with open("./new/test-gpt4omini/inferred.json", "r") as f:
    with open("./new/test-deepseekr1-distill-llama8b-recom/inferred.json", "r") as f:
        inferred = json.load(f)

    index_store = "./new/db/expl/texts.index"
    tweet_store = "./new/db/expl/texts.npy"
    explanation_store = "./new/db/expl/explanations.npy"
    sentiment_store = "./new/db/expl/sentiments.npy"
    vs = VectorStore(index_store=index_store, tweet_store=tweet_store, explanation_store=explanation_store, sentiment_store=sentiment_store, rebuild_if_exists=True)

    count = 0
    for index, item in tqdm(enumerate(inferred), desc="Building Vector Store", total=len(inferred)):

        tweet = item["text"]
        explanation = item["response"]
        predicted = item["predicted"]

        matching, distance, expl, text, sentiment = vs.store_new_text(explanation, tweet, predicted, similarity_threshold=0.7)

        if matching:
            count += 1

    print(f"Number of matches found: {count}")
    print(f"Length of vector store is {len(vs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--similarity', type=float, required=False)
    program_args = parser.parse_args()
    SIMILARITY = program_args.similarity
    
    main()
    print("Exiting Script.")