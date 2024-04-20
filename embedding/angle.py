import torch
from angle_emb import AnglE
from Config import Config


class EmbedderAnglE:
    def __init__(self, config: Config) -> None:
        if config.features["file_input"]["embed_config"]["angle"]["device"] == "cuda":
            self.model = AnglE.from_pretrained(config.features["file_input"]["embed_config"]["angle"]["model"],
                                               pooling_strategy='cls').to("cuda")
        else:
            self.model = AnglE.from_pretrained(config.features["file_input"]["embed_config"]["angle"]["model"],
                                               pooling_strategy='cls').to("cpu")

    def encode(self, text: str | list[str]) -> torch.Tensor:
        return self.model.encode(text, to_numpy=True)
