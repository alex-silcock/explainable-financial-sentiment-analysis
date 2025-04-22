import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from bert_classifier import BERTClassifier
import json
from datasets import DatasetDict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def main():
    cache_dir = "./.datasets/.stock_market_tweet"
    dataset = DatasetDict.load_from_disk(cache_dir)
    dataset = dataset["test"]

    classifier = BERTClassifier(model_path="./models/bert-large-uncased-stock-market-tweet-baseline-input")
    sentiments = classifier.infer_texts_batch(dataset["text"], batch_size=16)

    out = []
    labels = []
    predictions = []

    for index, instance in tqdm(enumerate(dataset), desc="Inferring", total=len(dataset)):
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

    inferred_output_file = "./new/baseline/inferred.json"
    with open(inferred_output_file, "w+") as f:
        json.dump(out, f, indent=4)

    metric_output_file = "./new/baseline/metrics.json"
    with open(metric_output_file, "w+") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":

    main()
    print("Exiting Script.")

