import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from bert_classifier import BERTClassifier
import json
from datasets import DatasetDict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Load model directly
from transformers import pipeline


def main():
    cache_dir = "./.datasets/.stock_market_tweet"
    dataset = DatasetDict.load_from_disk(cache_dir)
    dataset = dataset["test"]

    inputs = [instance["text"] for instance in dataset]
    
    pipe = pipeline("text-classification", model="ProsusAI/finbert", device="cuda")
    temp_sentiments = pipe(inputs)
    print(temp_sentiments[:3])

    sentiments = []
    for sentiment in temp_sentiments:
        _sentiment = -1
        if sentiment["label"] == "positive":
            _sentiment = 1
        elif sentiment["label"] == "negative":
            _sentiment = 0
        sentiments.append(_sentiment)

    print(sentiments[:3])

    out = []
    labels = []
    predictions = []

    for index, instance in tqdm(enumerate(dataset), desc="Inferring", total=len(dataset)):
        if sentiments[index] == -1:
            continue
        out.append({
            "text" : instance["text"],
            "sentiment" : instance["sentiment"],
            "predicted" : sentiments[index]
        })
        labels.append(instance["sentiment"])
        predictions.append(sentiments[index])


    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    metrics =  {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    print(metrics)

    inferred_output_file = "./new/baseline/finbert-inferred.json"
    with open(inferred_output_file, "w+") as f:
        json.dump(out, f, indent=4)

    metric_output_file = "./new/baseline/finbert-metrics.json"
    with open(metric_output_file, "w+") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":

    main()
    print("Exiting Script.")

