import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from lime.lime_text import LimeTextExplainer
import pandas as pd
import csv
from bert_classifier import BERTClassifier
import json
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Bearish", "Bullish"]

# model_map = {
#   response path : (model path, model name)
# }
model_map = {
    "./new/test-mistral7b-sentiment/responses_cleaned.json": ("./models/bert_mistral7b_finetune_with_sentiment", "Mistral 7B (Sentiment)"),
    "./new/test-deepseekr1-distill-llama8b-recom/responses_cleaned.json": ("./models/bert-deepseekr1-distill-llama8b-recom-finetune", "DeepSeek R1 (Distill LLaMA 8B)"),
    "./new/test-mistral7b/responses_cleaned.json": ("./models/bert-mistral7b-finetune", "Mistral 7B"),
    "./new/test-mistral7b/responses_with_input_cleaned.json": ("./models/bert_mistral7b_finetune_with_input", "Mistral 7B (Input)"),
    "./new/test-deepseekr1-distill-llama8b-recom/responses_with_input_cleaned.json": ("./models/bert-deepseekr1-distill-llama8b-recom-finetune-with-input", "DeepSeek R1 (Distill LLaMA 8B) (Input)"),
    "./new/test-mistral7b-finetuned/responses_cleaned.json": ("./models/bert_mistral7b_finetune_lora_finetuned", "Mistral 7B (Finetuned)"),
    "./new/test-deepseekr1-distill-qwen32b/responses_cleaned.json": ("./models/bert-deepseekr1-distill-qwen32b-recom-finetune", "DeepSeek R1 (Distill Qwen 32B)"),
    "./new/test-deepseekr1-distill-qwen32b/responses_with_input_cleaned.json": ("./models/bert-deepseekr1-distill-qwen32b-recom-finetune-with-input", "DeepSeek R1 (Distill Qwen 32B) (Input)"),
    "./new/test-gpt4omini/responses_cleaned.json": ("./models/bert_gpt4omini_finetune", "GPT4oMini"),
}

with open("./new/test-gpt4omini/inferred.json", "r") as f:
    responses = json.load(f)

bearish_responses = []
bullish_responses = []

for response in responses:
    if response['sentiment'] == 0:
        bearish_responses.append(response)
    elif response['sentiment'] == 1:
        bullish_responses.append(response)

sample_bearish = random.sample(bearish_responses, 15)
sample_bullish = random.sample(bullish_responses, 15)

sample_responses = sample_bearish + sample_bullish

frandom.shuffle(sample_responses)

with open("./new/lime_explanations/sample_responses.json", "w") as f:
    json.dump(sample_responses, f, indent=4)

results = []

for response_file, tup in model_map.items():
    model_path, model_label = tup
    print(f"Processing model: {model_path}")

    bert = BERTClassifier(model_path=model_path)
    model = bert.model
    tokenizer = bert.tokenizer

    for index, sample_text in enumerate(sample_responses):
        sample_text = sample_text.get("response")
        
        def predict_proba(texts):
            with torch.no_grad():
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding="longest",
                    # padding="max_length",
                    # max_length=512
                ).to(device)
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                return probs.detach().cpu().numpy()
        
        
        explainer = LimeTextExplainer(class_names=class_names)
        
        explanation = explainer.explain_instance(
            sample_text,
            predict_proba,
            num_features=10,
            num_samples=500
        )
        
        features_expl = "; ".join([f"{feat}:{weight:.4f}" for feat, weight in explanation.as_list()])
        results.append({"Model": model_label, "Explanation": features_expl})
        if not os.path.exists(f"./new/lime_explanations/{model_label}"):
            os.makedirs(f"./new/lime_explanations/{model_label}")
        explanation.save_to_file(f"./new/lime_explanations/{model_label}/{index}.html")
    
df = pd.DataFrame(results)
print("Aggregated LIME Explanations:")
print(df.to_string(index=False))

csv_path = "./new/lime_explanations/summary.csv"
df.to_csv(csv_path, index=False)
print(f"Saved aggregated explanations to {csv_path}")