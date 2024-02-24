import together
from transformers import LlamaTokenizerFast

from Config import Config


class TogetherAI:
    def __init__(self, config: Config) -> None:
        self.together_client = together
        self.together_client.api_key = config.backend_config.api_key
        self.together_config = config.backend_config
        self.llm_parameters = config.llm_parameters

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
        self.together_client.Complete.create(prompt, self.together_config.model, max_tokens, stopstrings, temperature,
                                             top_p, top_k, reppenalty)
