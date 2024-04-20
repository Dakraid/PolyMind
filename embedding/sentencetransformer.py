import torch
from sentence_transformers import SentenceTransformer
from Config import Config


class EmbedderSentenceTransformer:
    def __init__(self, config: Config) -> None:
        if config.features["file_input"]["embed_config"]["angle"]["device"] == "cuda":
            self.model = SentenceTransformer(config.features["file_input"]["embed_config"]["sentence_transformer"]["model"]).to("cuda")
        else:
            self.model = SentenceTransformer(config.features["file_input"]["embed_config"]["sentence_transformer"]["model"]).to("cpu")

    def encode(self, text: str | list[str]) -> torch.Tensor:
        return self.model.encode(text)
