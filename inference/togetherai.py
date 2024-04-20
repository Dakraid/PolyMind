import together
import json
import random
import Config
import requests
import traceback
from transformers import LlamaTokenizerFast


class TogetherAI:
    def __init__(self, config: Config) -> None:
        self.together_client = together
        self.together_client.api_key = config.backend_config.api_key
        self.together_config = config.backend_config
        self.llm_parameters = config.llm_parameters
        self.ctx_length = config.ctx_length

    def tokenize(self, text: str):
        tokenizer = LlamaTokenizerFast.from_pretrained(self.together_config.tokenizer_model)
        encoded_input = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(encoded_input)
        return {"length": len(encoded_input), "tokens": tokens}

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
                + f"\n{beginsep} {username} {prompt} {endsep} {modelname}"
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

        response = self.together_client.Complete.create(prompt, self.together_config.model, max_tokens,
                                                        stopstrings, temperature, top_p, top_k,
                                                        reppenalty)

        content += response['output']['choices'][0]['text']

        memory.append(
            f"\n{beginsep} {username} {prompt.strip()}\n{endsep} {modelname} {content.strip()}{eos}"
        )

        yield [content, memory, self.tokenize(prompt)["length"]]
