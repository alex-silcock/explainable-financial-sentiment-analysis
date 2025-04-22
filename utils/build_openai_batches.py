import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import argparse
from dotenv import load_dotenv
load_dotenv("./.env")
API_KEY = os.getenv("OPENAI_API_KEY")

import json
from openai import OpenAI
client = OpenAI()
from explain import GPTExplainer
MODEL = "gpt-4o-mini"
explainer = GPTExplainer(MODEL)

from helpers import load_sys_prompt, num_tokens_from_messages


# max tokens per batch req
error_margin = 25_000
batch_limits = {
    "gpt-4o": 90_000,
    "gpt-4o-mini": 2_000_000 - 100_000,
    "gpt-4": 100_000,
    "gpt-3.5-turbo": 2_000_000
}

def build_batches(prompts):
    batches = []
    batch = []

    tokens_in_batch = 0

    for i in range(len(prompts)):
        prompt = prompts.get(str(i))
        messages = explainer._create_prompt(prompt)

        max_tokens = 1000
        method = "POST"
        url = "/v1/chat/completions"

        req = {}

        req["custom_id"] = f"request-{i}"
        req["method"] = method
        req["url"] = url
        req["body"] = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens
        }

        num_tokens = num_tokens_from_messages(messages, model=MODEL)

        if (tokens_in_batch + num_tokens > batch_limits[MODEL]):
            batches.append(batch)
            batch = []
            tokens_in_batch = 0

        batch.append(req)
        tokens_in_batch += num_tokens

        if i + 1  == NUM_SAMPLES:
            break

    if batch: # append last batch
        batches.append(batch)

    return batches


def build_and_post():
    sys_prompt = load_sys_prompt(sys_prompt_path=SYS_PROMPT_PATH)
    explainer.set_sys_prompt(sys_prompt)

    prompts_file = f"{PROMPT_DIR}/prompt_info.json"

    with open(prompts_file, "r") as f:
        prompt_info = json.load(f)

    prompts = {}
    for i in range(len(prompt_info)):
        prompts[str(i)] = prompt_info[str(i)].get("prompt")

    batches = build_batches(prompts)
    batch_requests = []

    # write all to files
    for i, batch in enumerate(batches):
        print(f"Batch {i} has {len(batch)} prompts.")

        batch_req_file = f"{OUTPUT_DIR}/batchinput-{i}.jsonl"
        with open(batch_req_file, "w+") as f:
            for j in range(len(batch)):
                json.dump(batch[j], f)
                f.write("\n")


    for i, batch in enumerate(batches):
        batch_req_file = f"{OUTPUT_DIR}/batchinput-{i}.jsonl"
        batch_input_file = client.files.create(
            file=open(batch_req_file, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        batch_request = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Batch request {i} for stock prediction explanations."
            }
        )

        print(f"Batch request {i} created with id {batch_request.id}")
        start_time = time.time()
        while True:
            completed = await_completion(i, batch_request.id) # await completion before moving to next batch
            if completed:
                break
            else:
                elapsed_time = time.time() - start_time
                # format elapsed time to HH:MM:SS
                elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print(f"Elapsed time: {elapsed_time}")
                time.sleep(60) # sleep for 1 minute before checking again

        batch_requests.append(batch_request)

    return batch_requests


def await_completion(index, batch_request_id):
    # check if output file exists in directory already
    if os.path.exists(f"{OUTPUT_DIR}/batchoutput-{index}.jsonl"):
        print(f"Batch {index} output already exists. Skipping.")
        return True

    batch = client.batches.retrieve(batch_request_id)
    if batch.failed_at:
        print(f"Batch {batch_request_id} failed.")

        return False
    
    output_file_id = batch.output_file_id
    if not output_file_id:
        print(f"Batch {batch_request_id} is still processing.")
        return False

    file_response = client.files.content(output_file_id)
    output_path = f"{OUTPUT_DIR}/batchoutput-{index}.jsonl"
    with open(output_path, "w") as f:
        f.write(file_response.text)

    print(f"Batch {index} output written to {output_path}")
    return True


def cache_batch_request(batch_requests):
    for i, batch_request in enumerate(batch_requests):
        batch_request_dict = batch_request.to_dict()
        with open(f"{OUTPUT_DIR}/batchreq-{i}.json", "w") as f:
            json.dump(batch_request_dict, f, indent=4)
        print(f"Batch request {i} cached to {OUTPUT_DIR}/batchreq-{i}.json")


def load_cached_requests():
    batch_requests_ids = []

    some_random_max_number_of_batches = 1000
    for i in range(some_random_max_number_of_batches):
        try:
            with open(f"{OUTPUT_DIR}/batchreq-{i}.json", "r") as f:
                batch_request = json.load(f)
                batch_requests_ids.append(batch_request.get("id"))
        except FileNotFoundError:
            break

    return batch_requests_ids

def main():
    # TODO - need to add support for failed requests, and retry them

    batch_requests = build_and_post()
    cache_batch_request(batch_requests)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prompt_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, required=False)
    parser.add_argument('--sys_prompt_path', type=str, required=True)

    program_args = parser.parse_args()
    PROMPT_DIR = program_args.prompt_dir
    OUTPUT_DIR = program_args.output_dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    NUM_SAMPLES = program_args.num_samples
    SYS_PROMPT_PATH = program_args.sys_prompt_path

    main()
    print("Exiting Script.")