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
                if filename.endswith("aggregated_metrics.json"):
                    metrics_files.append(os.path.join(dirpath, filename))
    return metrics_files

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_all_metrics(metrics_data):

    rows = []
    for file_path, metrics, initial_metrics in metrics_data:
        for k_str, m in metrics.items():
            try:
                k_val = int(k_str)
            except ValueError:
                continue
            row = {"k": k_val}
            diff = {}
            for k, v in m.items():
                diff[k] = v - initial_metrics[k]

            row.update(diff)
                
            source = file_path.split("/")[-2]
    #                             Mistral 7B (sentiment)
    #         DeepSeek R1 (Distill LLaMA 8B)
    #                             Mistral 7B
    #                     Mistral 7B (Input)
    # DeepSeek R1 (Distill LLaMA 8B) (Input)
    #         Mistral 7B (FT, Dynamic Prompt)
    #                 Mistral 7B (Finetuned)
    #         DeepSeek R1 (Distill Qwen 32B)
    # DeepSeek R1 (Distill Qwen 32B) (Input)
    #                             GPT4oMini

            if "input" in file_path:
                source += "-input"
                
            if source == "test-mistral7b":
                source = "Mistral 7B"
            if source == "test-mistral7b-input":
                source = "Mistral 7B (Input)"
            if source == "test-mistral7b-sentiment":
                source = "Mistral 7B (sentiment)"
            if source == "test-gpt4omini":
                source = "GPT4oMini"
            if source == "dynamic-prompt-test":
                source = "Mistral 7B (FT, Dynamic Prompt)"
            if source == "test-deepseekr1-distill-llama8b-recom":
                source = "DeepSeek R1 (Distill LLaMA 8B)"
            if source == "test-deepseekr1-distill-llama8b-recom-input":
                source = "DeepSeek R1 (Distill LLaMA 8B) (Input)"
            if source == "test-deepseekr1-distill-qwen32b-recom":
                source = "DeepSeek R1 (Distill Qwen 32B)"
            if source == "test-deepseekr1-distill-qwen32b-recom-input":
                source = "DeepSeek R1 (Distill Qwen 32B) (Input)"
            if source == "test-mistral7b-finetuned":
                source = "Mistral 7B (Finetuned)"

            row["source"] = source
            rows.append(row)
    return pd.DataFrame(rows)

def flag_outliers(metrics_df):
    # Identify metric columns (skip "k" and "source")
    metric_columns = metrics_df.columns.difference(["k", "source"])
    for col in metric_columns:
        std_val = metrics_df[col].std()
        # Flag outliers: absolute difference greater than two standard deviations.
        metrics_df[col + '_outlier'] = metrics_df[col].abs() > 2 * std_val
    return metrics_df

def plot_metrics(metrics_df):
    plt.figure(figsize=(8,6))
    grouped = metrics_df.groupby("k")["f1"].mean().reset_index()
    plt.bar(grouped["k"].astype(str), grouped["f1"], color='skyblue')
    plt.xlabel("k")
    plt.ylabel("Mean F1")
    plt.title("Mean F1 Score vs k")
    plt.tight_layout()
    plt.show()
    plt.savefig("./new/plot/mean_f1_vs_k.png")

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
            initial_metrics = load_metrics(file_path.replace("aggregated_metrics.json", "metrics.json"))
            metrics_data.append((file_path, metrics, initial_metrics))
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
    
    if metrics_data:
        metrics_df = load_all_metrics(metrics_data)
        # metrics_df = flag_outliers(metrics_df)

        print("Aggregated Metrics DataFrame:")
        print(metrics_df)

        plot_metrics(metrics_df)
    else:
        print("No valid metrics data found.")

if __name__ == "__main__":
    main()
    print("Exiting Script.")