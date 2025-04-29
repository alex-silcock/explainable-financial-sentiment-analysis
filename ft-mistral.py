# Reference:
#
# This script has been taken and adapted from:
# https://medium.com/@harsh.vardhan7695/fine-tuning-llama-2-using-lora-and-qlora-a-comprehensive-guide-fd2260f0aa5f
#
# to work for Mistral 7B and our chosen dataset.
#
# I thank the author Harsh Vardan for his work.


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import ( 
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    
    )
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import DatasetDict, Dataset
from prompts.create_prompt import create_mistral_prompt
from utils.helpers import load_sys_prompt

torch.cuda.empty_cache()

# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = "./models/mistral-ai-7b-instruct"
cache_dir = "./.datasets/.cnbc_news_summarised_sentiment"
new_model = "./models/mistral-7b-explain-ft"

lora_r = 64         #lora attention dimension/ rank
lora_alpha = 16     #lora scaling parameter
lora_dropout = 0.1  #lora dropout probability

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

output_dir = "/app/results"

num_train_epochs = 10
fp16 = True
bf16 = False

per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 2
learning_rate = 2e-4
weight_decay = 0.001
max_grad_norm = 0.3
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
seed = 1
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 50
max_seq_length = 512
packing = False
device_map = {"":0}
dataset = Dataset.load_from_disk(cache_dir)

train_test_split = dataset.train_test_split(test_size=0.2)
test_valid_split = train_test_split["test"].train_test_split(test_size=0.5)

dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": test_valid_split["test"],
    "test": test_valid_split["train"]
})

print(dataset)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit = use_4bit,
    bnb_4bit_quant_type = bnb_4bit_quant_type,
    bnb_4bit_compute_dtype = compute_dtype,
    bnb_4bit_use_double_quant = use_nested_quant,)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map = device_map,
    torch_dtype=torch.float32,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

sys_prompt_path = "./new/sys_prompt_t.txt"
sys_prompt = load_sys_prompt(sys_prompt_path=sys_prompt_path)

def preprocess_function(examples):
    formatted_texts = []
    for article, summarisation in zip(examples["title"], examples["short_description"]):
        formatted_text = create_mistral_prompt(system_prompt=sys_prompt, user_message=article, assistant_message=summarisation)
        formatted_texts.append(formatted_text)
    return {"formatted_text": formatted_texts}

formatted_dataset = dataset.map(preprocess_function, batched=True, num_proc=8)

def tokenize_function(examples):
    return tokenizer(examples["formatted_text"], truncation=True, padding="max_length", max_length=2048)

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, num_proc=8)

print(tokenized_dataset)

peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r  = lora_r,
    bias = "none",
    task_type = "CAUSAL_LM",
)
training_arguments = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
    fp16 = fp16,
    bf16 = bf16,
    max_grad_norm = max_grad_norm,
    weight_decay = weight_decay,
    lr_scheduler_type = lr_scheduler_type,
    warmup_ratio = warmup_ratio,
    group_by_length = group_by_length,
    max_steps = max_steps,
    report_to = "tensorboard",
)
print("Set Training Args")

trainer = SFTTrainer(
    model = model,
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["validation"],
    peft_config = peft_config,
    args = training_arguments,
    tokenizer = tokenizer,
)
print("Create SFTTrainer")

print("Starting Training")
trainer.train()
print("Training Ended")

trainer.model.save_pretrained(new_model)
print("Saved pretrained model")
logging.set_verbosity(logging.CRITICAL)

instance_0 = tokenized_dataset["test"][0]
prompt = create_mistral_prompt(sys_prompt, instance_0["short_description"])
print(prompt)

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
result = pipe(prompt)

print(result[0]['generated_text'])

print("Merging model with LoRA weights")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"