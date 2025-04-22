from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_scheduler, EarlyStoppingCallback, TrainerCallback
import os
import matplotlib.pyplot as plt
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
device_map = {"":0}

class LossPlotCallback(TrainerCallback):
    def __init__(self, log_file="loss_log.txt"):
        self.losses = []
        self.log_file = log_file
        # Initialize the log file
        with open(self.log_file, "w") as f:
            f.write("Step,Loss\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            self.losses.append(loss)
            # Write the current step and loss to the log file.
            step = state.global_step
            with open(self.log_file, "a") as f:
                f.write(f"{step},{loss}\n")
        return control

def main():
    cache_dir = DATASET
    dataset = Dataset.load_from_disk(cache_dir)
    model_name = "bert-large-uncased" # base model to finetune

    model_save_dir = MODEL_PATH
    output_file = OUTPUT_METRIC_FILE

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        device_map=device_map
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def preprocess_function(examples):
        # return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        return tokenizer(examples["text"], truncation=True, padding="longest")


    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # create train validation and test splits

    train_val_dataset = tokenized_dataset.train_test_split(test_size=0.20)
    train_dataset = train_val_dataset['train']
    temp_dataset = train_val_dataset['test']

    val_test_dataset = temp_dataset.train_test_split(test_size=0.5)
    validation_dataset = val_test_dataset['train']
    test_dataset = val_test_dataset['test']

    print(len(train_dataset), len(validation_dataset), len(test_dataset))

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def compute_metrics(pred):
        logits, labels = pred
        predictions = torch.argmax(torch.tensor(logits), axis=-1)
        predictions = predictions.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        num_train_epochs=35,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_steps=500,
        eval_steps=500,
        save_steps=500
    )

    output_log_file = output_file.replace("./models", "")
    output_log_file = output_log_file.replace(".txt", "_log.txt")
    loss_callback = LossPlotCallback(log_file=output_log_file)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), loss_callback]
    )

    trainer.train()

    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(loss_callback.losses, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Logging Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    output_training_loss_file = output_file.replace("txt", "png")
    output_training_loss_file = output_training_loss_file.replace("./models", "")
    plt.savefig(output_training_loss_file)
    plt.show()

    test_results = trainer.predict(test_dataset)
    test_metrics = compute_metrics((test_results.predictions, test_results.label_ids))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        f.write("Test Set Evaluation Metrics\n")
        f.write("==========================\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"Metrics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-o', '--output_metric_file', type=str, required=True)
    program_args = parser.parse_args()
    
    DATASET = program_args.dataset
    MODEL_PATH = program_args.model_path
    OUTPUT_METRIC_FILE = program_args.output_metric_file

    main()
    print("Exiting Script.")