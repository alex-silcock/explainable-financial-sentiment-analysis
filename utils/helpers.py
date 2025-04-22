import json
import tiktoken

def load_settings(file):
    with open(file, 'r') as f:
        settings = json.load(f)
    return settings

def load_sys_prompt(sys_prompt_path):
    with open(sys_prompt_path, "r") as file:
        return file.read()
    

def num_tokens_from_messages(messages, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    token_count = 0
    for message in messages:
        token_count += 4
        for key, value in message.items():
            token_count += len(encoding.encode(value))
            if key == "name":
                token_count -= 1
    token_count += 2
    return token_count