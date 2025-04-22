import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from dotenv import load_dotenv
load_dotenv("./.env")
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from templates.explainer import Explainer
from accelerate import Accelerator
from peft import PeftModel


import torch
API_KEY = os.getenv("OPENAI_API_KEY")
from openai import OpenAI

import asyncio

class PredictionExtraction(BaseModel):
    analysis: str
    direction: str

class GPTExplainer(Explainer):
    def __init__(self, model):
        self.model = model
        self.client = self._initialise_model()

    def _initialise_model(self):
        return OpenAI(api_key=API_KEY)
    
    def _create_prompt(self, user_message):
        messages=[
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_message}
        ]
        return messages
    
    def _create_reasoning_prompt(self, user_message):
        messages=[
            {"role": "user", "content": user_message}
        ]
        return messages
    
    def explain(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._create_prompt(prompt),
        )
        return response.choices[0].message.content

    def parsed_explain(self, prompt):
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=self._create_prompt(prompt),
            response_format=PredictionExtraction,
        )
        return response.choices[0].message.parsed
    
    def reasoning_explain(self, prompt):
        if self.model not in ("o1-mini", "o1"):
            raise Exception(f"Selected model '{self.model}' does not support reasoning explanations")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._create_reasoning_prompt(prompt)
        )

        return response.choices[0].message.content
    
    def _format_response(self, response, remove_special_chars=True):
        analysis, direction = None, None

        if "{'direction'" in response:
            analysis = response.split("{'direction'")[0]
            direction = "{'direction'" + response.split("{'direction'")[1]

        if remove_special_chars:
            special_chars = ["\n", "```json"]
            for char in special_chars:
                analysis = analysis.replace(char, "")

            special_chars = ["\n", " ", "`"]
            for char in special_chars:
                direction = direction.replace(char, "")

        try:
            direction = json.loads(direction.replace("'", '"'))
        except Exception as e:
            print(e)

        return analysis, direction
    

class MistralInstructExplainer(Explainer):
    def __init__(self, model_path,  is_lora_ft=False, lora_model_path="", load_in_b16=True):
        self.model_path = model_path
        self.is_lora_ft = is_lora_ft
        self.load_in_b16 = load_in_b16
        self.lora_model_path = lora_model_path

        self.accelerator = Accelerator()
        self.tokenizer, self.model = self._initialise_model()


    def _initialise_model(self):
        if self.is_lora_ft:
            if not self.lora_model_path:
                raise ValueError("Must specify LoRA model path if using LoRA fine-tuned model")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and self.load_in_b16 else torch.float32
            )

            model = PeftModel.from_pretrained(base_model, self.lora_model_path)
            model = model.merge_and_unload()

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path
            )

            print("Loaded and merged LoRA fine-tuned model.")

        if os.path.exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and self.load_in_b16 else torch.float32,
                device_map="auto"
            )
        else:
            print(f"Model not found, downloading from Hugging Face and saving to {self.model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
                # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
            )
            model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and self.load_in_b16 else torch.float32,
                device_map="auto"
            )
            # tokenizer = AutoTokenizer.from_pretrained(
            #     "mistralai/Mistral-7B-Instruct-v0.3"
            # )
            # model = AutoModelForCausalLM.from_pretrained(
            #     "mistralai/Mistral-7B-Instruct-v0.3", 
            #     torch_dtype=torch.bfloat16 if torch.cuda.is_available() and self.load_in_b16 else torch.float32,
            #     # device_map="auto"
            # )
            model.save_pretrained(self.model_path)
            tokenizer.save_pretrained(self.model_path)

        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = self.accelerator.prepare(model)

        return tokenizer, model
    
    def _create_prompt(self, user_message):
        messages = [
            {"role": "system", "content": self.sys_prompt},
            # {"role": "user", "content": self.sys_prompt + " " + user_message + " <think>\n"} # if using deepseek model
            {"role": "user", "content": user_message}
        ]
        return messages
    
    async def explain_batch_async(self, instances, max_new_tokens=1000):
        batch_prompts = [self._create_prompt(instance) for instance in instances]

        inputs = self.tokenizer.apply_chat_template(
            batch_prompts,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded_responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        torch.cuda.empty_cache()

        return decoded_responses
    
    async def explain_batch_relevance_async(self, instances, max_new_tokens=1000):
        batch_prompts = [self._create_prompt(instance) for instance in instances]

        inputs = self.tokenizer.apply_chat_template(
            batch_prompts,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        just_responses = [response.split("relevant_stocks")[2] for response in responses]
        # just_responses = responses
        torch.cuda.empty_cache()

        return just_responses
