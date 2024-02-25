from typing import List

import numpy as np
import torch
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer
from transformers import ClapModel


class CLAPSimilarity:
    model: ClapModel
    tokenizer: AutoTokenizer
    feature_extractor: AutoFeatureExtractor

    def __init__(
        self,
        model_name="laion/larger_clap_general",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 48000,
    ):
        self.device = device
        self._init_model(model_name, device)
        self.sample_rate = sample_rate

    def _init_model(self, model_name: str, device: str):
        print(f"Initializing CLAP model: {model_name}")
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.model.eval()

        print(f"Initializing CLAP tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Initializing CLAP feature extractor: {model_name}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def get_text_embedding(self, texts: List[str]) -> torch.Tensor:
        features = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return self.model.get_text_features(**features)

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = [audio[i] for i in range(audio.shape[0])]
        else:
            audio = [audio]

        return self.feature_extractor(
            audio,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        )

    def get_audio_embedding(self, audio: np.ndarray) -> torch.Tensor:
        audio_features = self._preprocess_audio(audio)
        return self.model.get_audio_features(**audio_features)

    def compute_similarity(
        self, audio: np.ndarray, target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            audio_embeddings = self.get_audio_embedding(audio)
            if target_embeddings.device != audio_embeddings.device:
                target_embeddings = target_embeddings.to(audio_embeddings.device)

        similarity = cosine_similarity_matrix(audio_embeddings, target_embeddings)
        return similarity


def cosine_similarity_matrix(a, b):
    numerator = torch.einsum("bd,cd->bc", a, b)
    denominator = torch.norm(a[:, None], dim=-1) * torch.norm(b[None], dim=-1)
    return numerator / denominator


def test_clap():
    clap = CLAPSimilarity()

    targets = clap.get_text_embedding(
        [
            "King",
            "Octopus",
            "There is no spoon",
            "The cake is a lie",
            "Now this is a story all about how my life got flipped turned upside down",
        ],
    )

    fake_audios = np.random.normal(0, 1, (10, 100))
    similarity = clap.compute_similarity(fake_audios, targets)

    print(similarity.shape)  # (5, 10)
    print(similarity)


if __name__ == "__main__":
    test_clap()
