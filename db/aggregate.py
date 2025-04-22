import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from vector_store import VectorStore
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np


def load_initial_metrics(test_inferred_path):
    metric_path = test_inferred_path.replace("inferred.json", "metrics.json")
    with open(metric_path, "r") as f:
        inital_metrics = json.load(f)

    return inital_metrics


def build_vs(vector_store, inferred, similarity_threshold=0.7):
    for index, item in tqdm(enumerate(inferred), desc="Building Vector Store", total=len(inferred)):

        tweet = item["text"]
        explanation = item["response"]
        predicted = item["predicted"]

        _ = vector_store.store_new_text(explanation, tweet, predicted, similarity_threshold=similarity_threshold)


def build(train_inferred_path, test_inferred_path, similarity_threshold=0.7, k_nearest=5):

    inital_metrics = load_initial_metrics(test_inferred_path)

    with open(train_inferred_path, "r") as f:
        train_inferred = json.load(f)

    with open(test_inferred_path, "r") as f:
        test_inferred = json.load(f)


    base_path = "./new/db/agg/"
    if "input" in train_inferred_path:
        base_path = "./new/db/agg/input-"

    train_index_store = base_path + train_inferred_path.split("/")[2] + "/texts.index"
    train_tweet_store = base_path + train_inferred_path.split("/")[2] + "/texts.npy"
    train_explanation_store = base_path + train_inferred_path.split("/")[2] + "/explanations.npy"
    train_sentiment_store = base_path + train_inferred_path.split("/")[2] + "/sentiments.npy"

    test_index_store = base_path + test_inferred_path.split("/")[2] + "/texts.index"
    test_tweet_store = base_path + test_inferred_path.split("/")[2] + "/texts.npy"
    test_explanation_store = base_path + test_inferred_path.split("/")[2] + "/explanations.npy"
    test_sentiment_store = base_path + test_inferred_path.split("/")[2] + "/sentiments.npy"

    train_vs = VectorStore(index_store=train_index_store,
                           tweet_store=train_tweet_store, 
                           explanation_store=train_explanation_store, 
                           sentiment_store=train_sentiment_store)
    
    test_vs = VectorStore(index_store=test_index_store,
                           tweet_store=test_tweet_store, 
                           explanation_store=test_explanation_store, 
                           sentiment_store=test_sentiment_store)
    
    
    if len(train_vs) == 0:
        print("Building Train Vector Store")
        build_vs(train_vs, train_inferred, similarity_threshold=similarity_threshold)

    print(f"Number of instances in train vs: {len(train_vs)}")

    if len(test_vs) == 0:
        print("Building Test Vector Store")
        build_vs(test_vs, test_inferred, similarity_threshold=similarity_threshold)
    print(f"Number of instances in test vs: {len(test_vs)}")

    X_train, Y_train = build_features(train_inferred, train_vs, k_nearest, train=True)
    X_test, Y_test = build_features(test_inferred, test_vs, k_nearest)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, Y_train)

    test_pred = rf_classifier.predict(X_test)
    eval_metrics = evaluate(test_pred, Y_test)

    print("Initial Metrics:")
    print(inital_metrics)

    print("Evaluation Metrics:")
    print(eval_metrics)

    return eval_metrics

def build_features(inferred, vs, k_nearest, train=False):
    features = []
    labels = []

    for item in inferred:
        # find similar expl's
        d, expls, _, _, embedding, nearest_embeddings = vs.find_nearest(item["response"], k=k_nearest)

        if len(expls) == 0:
            print("No similar explanations found.")

        if train:
            sents = [item["sentiment"]] # gt
        else:
            sents = [item["predicted"]] # predicted

        flattened_embedding = embedding.flatten().tolist()
        nearest_arr = np.array([e.flatten() for e in nearest_embeddings])
        avg_nearest_embedding = np.mean(nearest_arr, axis=0).tolist()

        for i in range(len(expls)):
            for t_item in inferred:
                if expls[i] == t_item["response"]:
                    if train:
                        sents.append(t_item["sentiment"]) # gt
                        break
                    else:
                        sents.append(t_item["predicted"]) # predicted
                        break

        
        feature_vector = sents + flattened_embedding + avg_nearest_embedding
        features.append(feature_vector)
        labels.append(item["sentiment"]) # ground truth

    return features, labels

def evaluate(pred, gt):
    precision, recall, f1, _ = precision_recall_fscore_support(pred, gt, average="weighted")
    acc = accuracy_score(pred, gt)
    metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    return metrics


def main():

    models = [
        # ("./new/train-mistral7b/inferred.json", "./new/test-mistral7b/inferred.json"),
        # ("./new/train-mistral7b/responses_with_input_cleaned_with_input_inferred.json", "./new/test-mistral7b/responses_with_input_cleaned_with_input_inferred.json"),
        # ("./new/train-mistral7b-sentiment/inferred.json", "./new/test-mistral7b-sentiment/inferred.json"),
        # ("./new/train-mistral7b-finetuned/inferred.json", "./new/test-mistral7b-finetuned/inferred.json"),
        # ("./new/train-mistral7b-finetuned/inferred.json", "./new/dynamic-prompt-test/inferred.json"),
        # ("./new/train-deepseekr1-distill-llama8b-recom/inferred.json", "./new/test-deepseekr1-distill-llama8b-recom/inferred.json"),
        ("./new/train-deepseekr1-distill-llama8b-recom/responses_with_input_cleaned_with_input_inferred.json", "./new/test-deepseekr1-distill-llama8b-recom/responses_with_input_cleaned_with_input_inferred.json"),
        # ("./new/train-deepseekr1-distill-qwen32b-recom/inferred.json", "./new/test-deepseekr1-distill-qwen32b-recom/inferred.json"),
        ("./new/train-deepseekr1-distill-qwen32b-recom/responses_with_input_cleaned_with_input_inferred.json", "./new/test-deepseekr1-distill-qwen32b-recom/responses_with_input_cleaned_with_input_inferred.json"),
        # ("./new/train-gpt4omini/inferred.json", "./new/test-gpt4omini/inferred.json"),
    ]

    k_metric_list = {}

    for model in models:
        print(f"Running for model={model}")
        train_inferred_path, test_inferred_path = model

        for k in [1, 3, 5, 7]:
            print(f"Running for k={k}")
            eval_metrics = build(train_inferred_path=train_inferred_path,
                test_inferred_path=test_inferred_path,
                similarity_threshold=0.7, 
                k_nearest=k) # should update to not load vector store into memory each time
            
            k_metric_list[k] = eval_metrics

        with open(test_inferred_path.replace("inferred.json", "aggregated_metrics.json"), "w") as f:
            json.dump(k_metric_list, f, indent=4)
    


if __name__ == "__main__":
   
    main()
    print("Exiting Script.")