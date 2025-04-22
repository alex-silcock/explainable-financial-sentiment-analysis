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
    
    if TRAIN:
        dataset = dataset["train"]
        print("Inferring Train")
    else:
        dataset = dataset["test"]
        print("Inferring Test")

    with open(RESPONSE_FILE, "r") as f:
        responses = json.load(f)

    if INCLUDE_INPUT:
        responses_for_classification = ["Input: " + dataset[i]["text"] + " Response: " + responses[str(i)].get("response") for i in range(len(responses))]
    else:
        responses_for_classification = [responses[str(i)].get("response") for i in range(len(responses))]

    classifier = BERTClassifier(model_path=MODEL_PATH)
    sentiments = classifier.infer_texts_batch(responses_for_classification, batch_size=16)

    min_index = min(len(sentiments), len(dataset)) - 1

    out = []
    labels = []
    predictions = []

    for index, instance in tqdm(enumerate(dataset), desc="Inferring", total=len(dataset)):
        out.append({
            "text" : instance["text"],
            "response": responses[str(index)].get("response"),
            "sentiment" : instance["sentiment"],
            "predicted" : sentiments[index]
        })
        labels.append(instance["sentiment"])
        predictions.append(sentiments[index])

        if index == min_index:
            break

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    metrics =  {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    print(metrics)

    if INCLUDE_INPUT:
        inferred_output_file = RESPONSE_FILE.split(".json")[0] + "_with_input_inferred.json"
    else:
        inferred_output_file = "/".join(RESPONSE_FILE.split("/")[:-1]) + "/inferred.json"
    with open(inferred_output_file, "w+") as f:
        json.dump(out, f, indent=4)

    if INCLUDE_INPUT:
        metric_output_file = RESPONSE_FILE.split(".json")[0] + "_with_input_metrics.json"
    else:
        metric_output_file = "/".join(RESPONSE_FILE.split("/")[:-1]) + "/metrics.json"
    with open(metric_output_file, "w+") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--response_file', type=str, required=True)
    parser.add_argument('-i', '--include_input', type=str, required=False)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-t', '--train', type=str, required=False)
    program_args = parser.parse_args()
    
    RESPONSE_FILE = program_args.response_file
    INCLUDE_INPUT = program_args.include_input
    MODEL_PATH = program_args.model_path
    TRAIN = program_args.train


    main()
    print("Exiting Script.")

