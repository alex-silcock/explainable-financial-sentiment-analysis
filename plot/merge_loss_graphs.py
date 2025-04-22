import os
import math
import csv
import matplotlib.pyplot as plt

def find_log_files(root_directory):
    log_files = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.lower().endswith("_log.txt"):
                log_files.append(os.path.join(dirpath, filename))
    return log_files

def get_label(file_path):
    label_map = {
        "/app/outputs/other/bert-mistral7b-finetune_log.txt": "(a) Mistral 7B",
        "/app/outputs/other/bert-mistral7b-finetune-with-input_log.txt": "(b) Mistral 7B (Input)",
        "/app/outputs/other/bert-mistral7b-finetune-with-sentiment_log.txt": "(c) Mistral 7B (Sentiment)",
        "/app/outputs/other/bert-mistral7b-finetune-lora-finetuned_log.txt": "(d) Mistral 7B (Finetuned)",
        "/app/outputs/other/bert-gpt4omini-finetune_log.txt": "(e) GPT4oMini",
        "/app/outputs/other/bert-deepseekr1-distill-llama8b-recom-finetune_log.txt": "(f) DeepSeek R1 (Distill LLaMA 8B)",
        "/app/outputs/other/bert-deepseekr1-distill-llama8b-recom-finetune-with-input_log.txt": "(g) DeepSeek R1 (Distill LLaMA 8B) (Input)",
        "/app/outputs/other/bert-deepseekr1-distill-qwen32b-recom-finetune_log.txt": "(h) DeepSeek R1 (Distill Qwen 32B)",
        "/app/outputs/other/bert-deepseekr1-distill-qwen32b-recom-finetune-with-input_log.txt": "(i) DeepSeek R1 (Distill Qwen 32B) (Input)",
        "/app/outputs/other/bert-large-uncased-stock-market-tweet-baseline_log.txt": "(j) Baseline - BERT (Input)"
    }
    return label_map.get(file_path)

def main():
    root_directory = os.path.join(os.getcwd(), "outputs", "other")
    log_files = find_log_files(root_directory)
    
    num_images = len(log_files)
    n_cols = math.ceil(math.sqrt(num_images))
    n_rows = math.ceil(num_images / n_cols)

    n_cols = 3
    n_rows = 4
    
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    
    for index, file_path in enumerate(log_files):
        steps = []
        losses = []
        try:
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row:
                        steps.append(float(row[0]))
                        losses.append(float(row[1]))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        i = index + 1
        if file_path == "/app/outputs/other/bert-large-uncased-stock-market-tweet-baseline_log.txt":
            i += 1
            
        ax = plt.subplot(n_rows, n_cols, i)
        ax.plot(steps, losses, marker='o', linestyle='-')
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        label = get_label(file_path)
        ax.set_title(label, fontsize=12)
        ax.grid(True)
    
    plt.suptitle("Training Loss Graphs", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("./new/plot/merged_loss_graphs.png")
    plt.show()

if __name__ == "__main__":
    main()