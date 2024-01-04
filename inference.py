import json
import os
import random
from pathlib import Path
import Shared_vars
import requests
import traceback

API_ENDPOINT_URI = Shared_vars.API_ENDPOINT_URI
API_KEY = Shared_vars.API_KEY
TABBY = Shared_vars.TABBY
if TABBY:
    API_ENDPOINT_URI += "v1/completions"
else: 
    API_ENDPOINT_URI += "completion"

def tokenize(input):
    if TABBY:
        payload = {
        "add_bos_token": "true",
        "encode_special_tokens": "true",
        "decode_special_tokens": "true",
        "text": input
        }
        request = requests.post(
            API_ENDPOINT_URI.replace("completions", "token/encode"),
            headers={"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            json=payload,
            timeout=360,
        )
        return request.json()
    else:
        payload = {
        "content": input
        }
        request = requests.post(
            API_ENDPOINT_URI.replace("completion", "tokenize"),
            json=payload,
            timeout=360,
        )
        return {'length': len(request.json()['tokens']) }

def infer(prmpt, system='', temperature=0.7, username="", bsysep=Shared_vars.config.llm_parameters['bsysep'], esysep=Shared_vars.config.llm_parameters['esysep'], modelname="", eos="</s><s>",beginsep=Shared_vars.config.llm_parameters['beginsep'],endsep=Shared_vars.config.llm_parameters['endsep'], mem=[], few_shot="", max_tokens=250, stopstrings=[], top_p=1.0, top_k=Shared_vars.config.llm_parameters['top_k'], min_p=0.0):
    content = ''
    "<s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]"
    memory = mem
    prompt = f"{bsysep}\n"+ system + f"\n{esysep}\n" + few_shot + "".join(memory) + f"\n{beginsep} {username} {prmpt} {endsep} {modelname}" 
    #This feels wrong.

    print(f"Token count: {tokenize(prompt)['length']}")
    removal = 0
    while tokenize(prompt)['length'] + max_tokens / 2 > Shared_vars.config.ctxlen and len(memory) > 2:
        print(f"Removing old memories: Pass:{removal}")
        removal+=1
        memory = memory[removal:]
        prompt = f"{bsysep}\n"+ system + f"\n{esysep}\n" + few_shot + "".join(memory) + f"\n{beginsep} {username} {prmpt} {endsep} {modelname}" 

    payload = {
            "prompt": prompt,
            "model": "gpt-3.5-turbo-instruct",
            "max_tokens": max_tokens,
            "n_predict": max_tokens,
            "min_p": min_p,
            "stream": True,
            "seed": random.randint(1000002406736107, 3778562406736107), #Was acting weird without this
            "top_k": top_k,
            "top_p": top_p,
            "stop": [beginsep] + stopstrings + ["</s>", "<</SYS>>", "[INST]", "[/INST]"],
            "temperature": temperature,
        }

    request = requests.post(
        API_ENDPOINT_URI,
        headers={"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        json=payload,
        stream=True,
        timeout=360,
    )

    if request.encoding is None:
        request.encoding = "utf-8"

    for line in request.iter_lines(decode_unicode=True):
        if line:
            if TABBY:
                try:
                    print(json.loads(" ".join(line.split(" ")[1:]))['choices'][0]['text'], end="", flush=True)

                    content += json.loads(" ".join(line.split(" ")[1:]))['choices'][0]['text']
                except Exception as e:
                    print(e)

            else:
                try:
                    if "data" in line:
                        print(json.loads(" ".join(line.split(" ")[1:]))['content'], end="", flush=True)

                        content += json.loads(" ".join(line.split(" ")[1:]))['content']           
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
    print("")
    memory.append(f"\n{beginsep} {username} {prmpt.strip()}\n{endsep} {modelname} {content.strip()}{eos}")
    return [content,memory,tokenize(prompt)['length']]