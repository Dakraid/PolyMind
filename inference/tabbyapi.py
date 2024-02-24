import json
import random
import Config
import requests
import traceback


class TabbyAPI:
    def __init__(self, config: Config) -> None:
        self.compat = config.compat
        self.tabby_api = config.backend == "tabbyapi"
        self.backend_config = config.backend_config
        if self.tabby_api:
            self.backend_config.api_endpoint_uri += "v1/completions"
        else:
            self.backend_config.api_endpoint_uri += "completion"

        if self.compat:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.token_model)

        self.llm_parameters = config.llm_parameters
        self.ctx_length = config.ctx_length

    def tokenize(self, text):
        if self.compat:
            encoded_input = self.tokenizer.encode(text, return_tensors=None)
            tokens = self.tokenizer.convert_ids_to_tokens(encoded_input)
            return {"length": len(encoded_input), "tokens": tokens}
        else:
            if self.tabby_api:
                payload = {
                    "add_bos_token": "true",
                    "encode_special_tokens": "true",
                    "decode_special_tokens": "true",
                    "text": text,
                }
                request = requests.post(
                    self.backend_config.api_endpoint_uri.replace("completions", "token/encode"),
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.backend_config.api_key}",
                    },
                    json=payload,
                    timeout=360,
                )
                return request.json()
            else:
                payload = {"content": text}
                request = requests.post(
                    self.backend_config.api_endpoint_uri.replace("completion", "tokenize"),
                    json=payload,
                    timeout=360,
                )
                return {"length": len(request.json()["tokens"])}

    def infer(
            self,
            prompt,
            system="",
            temperature=0.7,
            username="",
            bsysep="",
            esysep="",
            modelname="",
            eos="</s><s>",
            beginsep="",
            endsep="",
            mem=None,
            few_shot="",
            max_tokens=250,
            stopstrings=[],
            top_p=1.0,
            top_k="",
            min_p=0.0,
            streamresp=False,
            reppenalty=1.0,
            max_temp=0,
            min_temp=0
    ):
        if mem is None:
            mem = []
        bsysep = self.llm_parameters.sys_begin_sep
        esysep = self.llm_parameters.sys_end_sep
        beginsep = self.llm_parameters.begin_sep
        endsep = self.llm_parameters.end_sep
        top_k = self.llm_parameters.top_k
        reppenalty = self.llm_parameters.repetition_penalty if reppenalty == 1.0 else reppenalty
        max_temp = self.llm_parameters.max_temp if max_temp == 0 else max_temp
        min_temp = self.llm_parameters.min_temp if min_temp == 0 else min_temp

        content = ""
        memory = mem
        prompt = (
                f"{bsysep}\n"
                + system
                + f"\n{esysep}\n"
                + few_shot
                + "".join(memory)
                + f"\n{beginsep} {username} {prompt}{endsep} {modelname}"
        )
        # This feels wrong.

        print(f"Token count: {self.tokenize(prompt)['length']}")
        removal = 0
        while (
                self.tokenize(prompt)["length"] + max_tokens / 2 > self.ctx_length
                and len(memory) > 2
        ):
            print(f"Removing old memories: Pass:{removal}")
            removal += 1
            memory = memory[removal:]
            prompt = (
                    f"{bsysep}\n"
                    + system
                    + f"\n{esysep}\n"
                    + few_shot
                    + "".join(memory)
                    + f"\n{beginsep} {username} {prompt} {endsep} {modelname}"
            )
        stopstrings += ["</s>", "<</SYS>>", "[Inst]", "[/INST]", self.llm_parameters.sys_begin_sep,
                        self.llm_parameters.sys_end_sep, self.llm_parameters.begin_sep,
                        self.llm_parameters.end_sep]
        payload = {
            "prompt": prompt,
            "model": "gpt-3.5-turbo-instruct",
            "max_tokens": max_tokens,
            "n_predict": max_tokens,
            "min_p": min_p,
            "repetition_penalty": reppenalty,
            "stream": True,
            "seed": random.randint(
                1000002406736107, 3778562406736107
            ),  # Was acting weird without this
            "top_k": top_k,
            "top_p": top_p,
            "stop": [beginsep] + stopstrings,
            "temperature": temperature,
        }
        if min_temp != 0 and max_temp != 0:
            payload["min_temp"] = min_temp
            payload["max_temp"] = max_temp
        request = requests.post(
            self.backend_config.api_endpoint_uri,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.backend_config.api_key}",
            },
            json=payload,
            stream=True,
            timeout=360,
        )

        if request.encoding is None:
            request.encoding = "utf-8"
        prevtoken = ""
        repetitioncount = 0
        for line in request.iter_lines(decode_unicode=True):
            if line:
                if self.tabby_api:
                    if " ".join(line.split(" ")[1:]) != "[DONE]":
                        if (
                                prevtoken
                                == json.loads(" ".join(line.split(" ")[1:]))["choices"][0][
                            "text"
                        ]
                        ):
                            repetitioncount += 1
                            if repetitioncount > 25:
                                print("Stopping loop due to repetition")
                                break
                        else:
                            repetitioncount = 0
                        prevtoken = json.loads(" ".join(line.split(" ")[1:]))["choices"][0][
                            "text"
                        ]
                        print(
                            json.loads(" ".join(line.split(" ")[1:]))["choices"][0]["text"],
                            end="",
                            flush=True,
                        )
                        if streamresp:
                            yield json.loads(" ".join(line.split(" ")[1:]))["choices"][0][
                                "text"
                            ]

                        content += json.loads(" ".join(line.split(" ")[1:]))["choices"][0][
                            "text"
                        ]

                else:
                    try:
                        if "data" in line:
                            print(
                                json.loads(" ".join(line.split(" ")[1:]))["content"],
                                end="",
                                flush=True,
                            )
                            if (
                                    prevtoken
                                    == json.loads(" ".join(line.split(" ")[1:]))["content"]
                            ):
                                repetitioncount += 1
                            if repetitioncount > 25:
                                print("Stopping loop due to repetition")
                                break
                            else:
                                repetitioncount = 0
                            prevtoken = json.loads(" ".join(line.split(" ")[1:]))["content"]
                            if streamresp:
                                yield json.loads(" ".join(line.split(" ")[1:]))["content"]

                            content += json.loads(" ".join(line.split(" ")[1:]))["content"]

                    except Exception:
                        print(traceback.format_exc())
        print("")
        memory.append(
            f"\n{beginsep} {username} {prompt.strip()}\n{endsep} {modelname} {content.strip()}{eos}"
        )

        yield [content, memory, self.tokenize(prompt)["length"]]
