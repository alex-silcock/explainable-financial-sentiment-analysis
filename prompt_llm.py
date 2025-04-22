import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{__file__} using device: {device}")

from explain import MistralInstructExplainer
import json
from ds_utils.PromptDataset import PromptDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from utils.helpers import load_sys_prompt

import argparse
import asyncio

async def process_batch(exp, batch, progress_bar, cache_file, responses_cache):
    prompts = [batch[i] for i in range(len(batch))]
    batch_responses = await exp.explain_batch_async(prompts)
    progress_bar.update(1)

    for i, response in enumerate(batch_responses):
        responses_cache.append(response)
    
    with open(cache_file, "w") as f:
        json.dump(responses_cache, f, indent=4)

    return batch_responses

async def batch_process_prompts_async(exp, prompts, batch_size=4, cache_file="./outputs/response_cache.json"):
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=2,
        collate_fn=lambda x: x
    )

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            responses_cache = json.load(f)
    else:
        responses_cache = []

    tasks = []
    with tqdm_asyncio(total=len(dataloader), desc="Processing Batches") as progress_bar:
        for batch in dataloader:
            tasks.append(process_batch(exp, batch, progress_bar, cache_file, responses_cache))

        all_responses = await asyncio.gather(*tasks)
    
    return [response for batch_responses in all_responses for response in batch_responses]


def main():
    with open(f"{PROMPT_DIR}/prompt_info.json", "r") as f:
        prompt_info = json.load(f)


    prompts, dates, tickers, contains_trump, contains_cnbc = [], [], [], [], []

    for i in range(len(prompt_info)):
        cur_prompt = prompt_info[str(i)]

        prompts.append(cur_prompt.get("prompt"))
        dates.append(cur_prompt.get("date"))
        tickers.append(cur_prompt.get("ticker"))
        contains_trump.append(cur_prompt.get("contains_trump"))
        contains_cnbc.append(cur_prompt.get("contains_cnbc"))

    if USE_MISTRAL_FT:
        exp = MistralInstructExplainer(model_path="./models/mistral-ai-7b-instruct", is_lora_ft=True, lora_model_path="./models/mistral-7b-explain-ft")#, load_in_b16=False)
    else:
        # exp = MistralInstructExplainer(model_path="models/DeepSeek-R1-Distill-Qwen-32B")#, load_in_b16=False)
        # exp = MistralInstructExplainer(model_path="./models/mistral-ai-7b-instruct")#, load_in_b16=False)
        exp = MistralInstructExplainer(model_path="./models/DeepSeek-R1-Distill-Llama-8B")#, load_in_b16=False)
        
    sys_prompt = load_sys_prompt(sys_prompt_path=SYS_PROMPT_PATH)
    exp.set_sys_prompt(sys_prompt)

    responses = asyncio.run(batch_process_prompts_async(exp, prompts, batch_size=4, cache_file=f"{OUTPUT_DIR}/response_cache.json"))

    resj = {}
    for i in range(len(responses)):
        temp = {}
        temp["ticker"] = tickers[i]
        temp["date"] = dates[i]
        temp["response"] = responses[i]
        temp["contains_trump"] = contains_trump[i]
        temp["contains_cnbc"] = contains_cnbc[i]

        resj[i] = temp

    with open(f"{OUTPUT_DIR}/responses.json", "w+") as f:
        json.dump(resj, f, indent=4)

    with open(f"{OUTPUT_DIR}/metadata.json", "w+") as f:
        out = {"sys_prompt_path": SYS_PROMPT_PATH}
        json.dump(out, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sys_prompt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prompt_dir', type=str, required=True)
    parser.add_argument('--use_mistral_ft', type=str, required=False)

    program_args = parser.parse_args()
    SYS_PROMPT_PATH = program_args.sys_prompt_path
    OUTPUT_DIR = program_args.output_dir

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    PROMPT_DIR = program_args.prompt_dir
    USE_MISTRAL_FT = program_args.use_mistral_ft

    main()
    print("Exiting Script.")
