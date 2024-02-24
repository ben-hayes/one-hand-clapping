from typing import List

import torch
from transformers import ClapModel, ClapProcessor


class CLAPSimilarity:
    model: ClapModel
    processor: ClapProcessor

    def __init__(
        self,
        model_name="laion/larger_clap_general",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self._init_model(model_name, device)

    def _init_model(self, model_name: str, device: str):
        print(f"Initializing CLAP model: {model_name}")
        self.model = ClapModel.from_pretrained(model_name).to(device)

        print(f"Initializing CLAP processor: {model_name}")
        self.processor = ClapProcessor.from_pretrained(model_name).to(device)

    def get_text_embedding(self, texts: List[str]) -> torch.Tensor:
        features = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return self.model.get_text_features(**features)

    def get_audio_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio[None, None]
        elif audio.ndim == 2:
            audio = audio[None]
        elif audio.ndim != 3:
            raise ValueError(f"Invalid audio shape: {audio.shape}")

        if audio.device != self.device:
            audio = audio.to(self.device)

        features = self.processor(
            audios=audio,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return self.model.get_audio_features(**features)

    def compute_similarity(
        self, audio: torch.Tensor, target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        audio_embeddings = self.get_audio_embedding(audio)
        if target_embeddings.device != audio_embeddings.device:
            target_embeddings = target_embeddings.to(audio_embeddings.device)
        return cosine_similarity_matrix(target_embeddings, audio_embeddings)


def cosine_similarity_matrix(a, b):
    numerator = torch.einsum("bd,cd->bc", a, b)
    denominator = torch.norm(a[:, None], dim=-1) * torch.norm(b[None], dim=-1)
    return numerator / denominator


def test_clap():
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    model = ClapModel.from_pretrained("laion/larger_clap_general")

    inputs = processor(
        text=[
            "King",
            "Prince",
            "Octopus",
            "Suffering",
            "Famine",
            "Now this is a story all about how my life got flipped turned upside down",
        ],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    print(inputs)
    outputs = model.get_text_features(**inputs)

    print(outputs)  # torch.Size([3, 2])
    print(cosine_similarity_matrix(outputs, outputs))  # torch.Size([3, 3])


if __name__ == "__main__":
    test_clap()
