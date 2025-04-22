import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def find_metrics_files(root_directory):
    metrics_files = []
    for dirpath, dirnames, filenames in os.walk(root_directory):

        if "test" in dirpath and "resp" not in dirpath:
            for filename in filenames:
                if filename.endswith("metrics.json") and not filename.endswith("aggregated_metrics.json"):
                    metrics_files.append(os.path.join(dirpath, filename))

    metrics_files.append("./new/baseline/metrics.json")
    metrics_files.append("./new/baseline/rf-metrics.json")
    metrics_files.append("./new/baseline/finbert-metrics.json")
    metrics_files.pop(metrics_files.index("/app/new/test-deepseekr1-distill-llama8b/metrics.json"))
    metrics_files.pop(metrics_files.index("/app/new/test-deepseekr1-distill-qwen32b/metrics.json"))
    return metrics_files

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return {
        'accuracy': data.get('accuracy', 0),
        'precision': data.get('precision', 0),
        'recall': data.get('recall', 0),
        'f1': data.get('f1', 0)
    }

def plot_metrics(metrics_data):
    path_file_labels = [os.path.relpath(fp, start=os.getcwd()) for fp, _ in metrics_data]

    file_labels = []

    for label in path_file_labels:
        if label == "new/test-mistral7b-finetuned/metrics.json":
            label = "Mistral 7B (Finetuned)"
            file_labels.append(label)
            continue

        if label == "new/test-deepseekr1-distill-llama8b-recom/metrics.json":
            label = "DeepSeek R1 (Distill LLaMA 8B)"
            file_labels.append(label)
            continue

        if label == "new/test-deepseekr1-distill-qwen32b-recom/metrics.json":
            label = "DeepSeek R1 (Distill Qwen 32B)"
            file_labels.append(label)
            continue

        if label == "new/test-deepseekr1-distill-llama8b-recom/responses_with_input_cleaned_with_input_metrics.json":
            label = "DeepSeek R1 (Distill LLaMA 8B) (Input)"
            file_labels.append(label)
            continue

        if label == "new/test-deepseekr1-distill-qwen32b-recom/responses_with_input_cleaned_with_input_metrics.json":
            label = "DeepSeek R1 (Distill Qwen 32B) (Input)"
            file_labels.append(label)
            continue

        if label == "new/test-deepseekr1-distill-qwen32b/metrics.json":
            label = "DeepSeek R1 (Distill Qwen 32B)"
            file_labels.append(label)
            continue

        if label == "new/dynamic-prompt-test/metrics.json":
            label = "Mistral 7B (FT, Dynamic Prompt)"
            file_labels.append(label)
            continue

        if "baseline" in label:
            if "rf" in label:
                label = "Baseline RF"
            elif "finbert" in label:
                label = "Baseline FinBERT"
            else:
                label = "Baseline (Input)"
            file_labels.append(label)
            continue

        label = label.replace("new/test-", "")
        if "input" in label:
            label = label.split("/")[0] + " (Input)"
        label = label.split("/")[0]
            
        if "-" in label:
            split = label.split("-")
            label = split [0] + " (" + split[1] + ")" 

        if "mistral7b" in label:
            label = label.replace("mistral7b", "Mistral 7B")
        if "gpt4omini" in label:
            label = label.replace("gpt4omini", "GPT4oMini")

        file_labels.append(label)

    accuracy = [m['accuracy'] for _, m in metrics_data]
    precision = [m['precision'] for _, m in metrics_data]
    recall = [m['recall'] for _, m in metrics_data]
    f1 = [m['f1'] for _, m in metrics_data]
    
    norm = plt.Normalize(0, 100)
    
    cmap = plt.cm.Blues

    colors_accuracy = [cmap(min(0.15 + norm(val)*0.4, 1)) for val in accuracy]
    colors_precision = [cmap(min(0.4 + norm(val)*0.3, 1)) for val in precision]
    colors_recall = [cmap(min(0.7 + norm(val)*0.2, 1)) for val in recall]
    colors_f1 = [cmap(min(0.9 + norm(val)*0.1, 1)) for val in f1]

    x = np.arange(len(file_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, len(file_labels)*2), 6))
    
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color=colors_accuracy)
    ax.bar(x - 0.5*width, precision, width, label='Precision', color=colors_precision)
    ax.bar(x + 0.5*width, recall, width, label='Recall', color=colors_recall)
    ax.bar(x + 1.5*width, f1, width, label='F1 Score', color=colors_f1)

    max_f1 = max(f1)
    ax.axhline(y=max_f1, color='red', linestyle='dotted', label='Max F1')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Metrics by model')
    ax.set_xticks(x)
    ax.set_xticklabels(file_labels, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    plt.savefig('./new/plot/metrics_plot.png')
    plt.show()

    csv_file = "./new/plot/metrics_summary.csv"
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1"])
        for lbl, acc, prec, rec, f1_score in zip(file_labels, accuracy, precision, recall, f1):
            writer.writerow([lbl, acc, prec, rec, f1_score])

    df = pd.DataFrame({
        "Model": file_labels,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })
    print("Aggregated Metrics:")
    print(df)

def main():
    root_directory = os.path.join(os.getcwd(), "new")
    metrics_files = find_metrics_files(root_directory)
    
    if not metrics_files:
        print("No metrics.json files found in the ./new directory.")
        return
    
    metrics_data = []
    for file_path in metrics_files:
        try:
            metrics = load_metrics(file_path)
            metrics_data.append((file_path, metrics))
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    metrics_data.sort(key=lambda x: x[1]["f1"]) # sort by f1 score
    
    if metrics_data:
        plot_metrics(metrics_data)
    else:
        print("No valid metrics data found.")

if __name__ == "__main__":
    main()
    print("Exiting Script.")