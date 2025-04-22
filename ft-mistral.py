import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from datasets import load_dataset
from transformers import ( 
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
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

#output directory where the model predictions and checkpoints will be stored
output_dir = "/app/results"

#number of training epochs
num_train_epochs = 10

#enable fp16/bf16 training (set bf16 to True when using A100 GPU in google colab)
fp16 = True
bf16 = False

#batch size per GPU for training
per_device_train_batch_size = 2

#batch size per GPU for evaluation
per_device_eval_batch_size = 2

#gradient accumulation steps - No of update steps
gradient_accumulation_steps = 2

#learning rate
learning_rate = 2e-4

#weight decay
weight_decay = 0.001

#Gradient clipping(max gradient Normal)
max_grad_norm = 0.3

#optimizer to use
optim = "paged_adamw_32bit"

#learning rate scheduler
lr_scheduler_type = "cosine"

#seed for reproducibility
seed = 1

#Number of training steps
max_steps = -1

#Ratio of steps for linear warmup
warmup_ratio = 0.03

#group sequnces into batches with same length
group_by_length = True

#save checkpoint every X updates steps
save_steps = 0

#Log at every X updates steps
logging_steps = 50

#maximum sequence length to use
max_seq_length = 512

packing = False

#load the entire model on the GPU
device_map = {"":0}

#load dataset
dataset = Dataset.load_from_disk(cache_dir)

# Split into train, validation, and test datasets with 80%, 10%, 10% split
train_test_split = dataset.train_test_split(test_size=0.2)
test_valid_split = train_test_split["test"].train_test_split(test_size=0.5)

dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": test_valid_split["test"],
    "test": test_valid_split["train"]
})

print(dataset)

# dataset["validation"] = dataset["test"]
# del dataset["test"]

#load tokenizer and model with QLoRA config
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit = use_4bit,
    bnb_4bit_quant_type = bnb_4bit_quant_type,
    bnb_4bit_compute_dtype = compute_dtype,
    bnb_4bit_use_double_quant = use_nested_quant,)

#cheking GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("="*80)
        print("Your GPU supports bfloat16, you are getting accelerate training with bf16= True")
        print("="*80)

#load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map = device_map,
    torch_dtype=torch.float32,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

#Load LLama tokenizer
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

# Apply preprocessing to the dataset
formatted_dataset = dataset.map(preprocess_function, batched=True, num_proc=8)

def tokenize_function(examples):
    return tokenizer(examples["formatted_text"], truncation=True, padding="max_length", max_length=2048)


tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, num_proc=8)

print(tokenized_dataset)

#Load QLoRA config
peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r  = lora_r,
    bias = "none",
    task_type = "CAUSAL_LM",
)

#Set Training parameters
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

#SFT Trainer
trainer = SFTTrainer(
    model = model,
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["validation"],
    peft_config = peft_config,
    # processing_class = "formatted_text",
    # max_seq_length = max_seq_length,
    args = training_arguments,
    tokenizer = tokenizer,
    # packing = packing,
)
print("Create SFTTrainer")

print("Starting Training")
trainer.train()
print("Training Ended")

#save trained model
trainer.model.save_pretrained(new_model)
print("Saved pretrained model")

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
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

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"