import time
from typing import Any, List, Union

import numpy as np
import torch
from mistralai.client import MistralClient
from numpy import ndarray, dtype
from torch import Tensor

import Config


class EmbedderMistralAI:
    def __init__(self, config: Config) -> None:
        self.mistral_client = MistralClient(api_key=config.features["file_input"]["embed_config"]["api"]["api_key"])
        self.model = config.features["file_input"]["embed_config"]["api"]["model"]

    def encode(self, text: str | list[str]) -> Union[List[Tensor], ndarray, Tensor]:
        output = []
        if text is str:
            embeddings_batch_response = self.mistral_client.embeddings(
                  model=self.model,
                  input=text,
              )
            output.append(embeddings_batch_response.data[0].embedding)
        else:
            counter = 0
            for chunk in text:
                counter += 1
                if counter >= 2:
                    time.sleep(1)
                    counter = 0

                embeddings_batch_response = self.mistral_client.embeddings(
                    model=self.model,
                    input=chunk,
                )
                output.append(embeddings_batch_response.data[0].embedding)

        return np.array(output)
