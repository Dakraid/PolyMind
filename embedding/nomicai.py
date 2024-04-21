import time
from typing import Any, List, Union

import nomic
import numpy as np
import torch
from nomic import embed
from numpy import ndarray, dtype
from torch import Tensor

import Config


class EmbedderNomicAI:
    def __init__(self, config: Config) -> None:
        self.apikey = config.features["file_input"]["embed_config"]["api"]["api_key"]
        self.model = config.features["file_input"]["embed_config"]["api"]["model"]

    def encode(self, text: str | list[str]) -> Union[List[Tensor], ndarray, Tensor]:
        output = []
        if text is str:
            nomic.login(self.apikey)
            embeddings_batch_response = nomic.embed.text(
                texts=text,
                model=self.model
            )
            output.append(embeddings_batch_response["embeddings"])
        else:
            counter = 0
            for chunk in text:
                counter += 1
                if counter >= 2:
                    time.sleep(1)
                    counter = 0

                nomic.login(self.apikey)
                embeddings_batch_response = nomic.embed.text(
                    texts=chunk,
                    model=self.model
                )
                output.append(embeddings_batch_response["embeddings"])

        return np.array(output)
