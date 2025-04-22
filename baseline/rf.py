import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from bert_classifier import BERTClassifier
import json
from datasets import DatasetDict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    cache_dir = "./.datasets/.stock_market_tweet"
    dataset = DatasetDict.load_from_disk(cache_dir)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_texts = [item["text"] for item in train_dataset]
    train_labels = [item["sentiment"] for item in train_dataset]
    
    test_texts = [item["text"] for item in test_dataset]
    test_labels = [item["sentiment"] for item in test_dataset]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(train_vectors, train_labels)

    sentiments = rf_classifier.predict(test_vectors)

    out = []
    labels = []
    predictions = []

    for index, instance in tqdm(enumerate(test_dataset), desc="Inferring", total=len(test_dataset)):
        out.append({
            "text": instance["text"],
            "sentiment": instance["sentiment"],
            "predicted": int(sentiments[index])
        })
        labels.append(instance["sentiment"])
        predictions.append(sentiments[index])

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    print(metrics)

    inferred_output_file = "./new/baseline/rf-inferred.json"
    with open(inferred_output_file, "w+") as f:
        json.dump(out, f, indent=4)

    metric_output_file = "./new/baseline/rf-metrics.json"
    with open(metric_output_file, "w+") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":

    main()
    print("Exiting Script.")

