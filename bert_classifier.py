import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from ds_utils.TextDataset import TextDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERTClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

        self.model, self.tokenizer = self._load_tokenizer_and_model()

        # self.sentiments = ["Negative", "Positive"]
        self.sentiments = [0, 1]

    def _load_tokenizer_and_model(self):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model.eval()
            return (model, tokenizer)
        except:
            raise Exception(f"Could not load tokenizer and model from path: {self.model_path}")
        
    def infer_texts(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        mapped_sentiments = [self.sentiments[pred] for pred in preds]
        return mapped_sentiments
       
    def infer_texts_batch(self, texts, batch_size=16):
        texts = list(texts) if not isinstance(texts, list) else texts

        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset,
            batch_size=batch_size,
            num_workers=2,
            collate_fn=lambda x: x
        )
        
        results = []
        
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for batch_idx, batch_texts in enumerate(tqdm(dataloader, desc="Inferring Sentiments")):
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512
                ).to(device)
                outputs = self.model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                preds = preds.cpu()
                mapped_sentiments = [self.sentiments[pred] for pred in preds]
                results.extend(mapped_sentiments)
                
                del inputs, outputs, probs, preds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # print(f"Finished batch {batch_idx + 1} of {len(dataloader)}") # if running as background process to check logs
        
        return results