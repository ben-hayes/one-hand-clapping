from typing import List, Callable

import numpy as np
import torch
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer
from transformers import ClapModel
from torchaudio.transforms import MelSpectrogram


class ParallelClapFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: int = 0,
        f_max: int = 24000,
        power: float = 2.0,
        window_fn: Callable = torch.hann_window,
        norm: str = "slaney",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            window_fn=window_fn,
            norm=norm,
        ).to(device)

    def __call__(
        self,
        raw_speech: torch.Tensor,
    ):
        input_mel = self.transform(raw_speech).transpose(-1, -2).unsqueeze(1)
        is_longer = [False] * input_mel.shape[0]

        input_features = {"input_features": input_mel, "is_longer": is_longer}

        return input_features


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
        fe = AutoFeatureExtractor.from_pretrained(model_name)

        self.feature_extractor = ParallelClapFeatureExtractor(
            fe.sampling_rate,
            fe.fft_window_size,
            fe.hop_length,
            fe.feature_size,
            fe.frequency_min,
            fe.frequency_max,
            2.0,
            torch.hann_window,
            "slaney",
            device=self.device,
        )

    def get_text_embedding(self, texts: List[str]) -> torch.Tensor:
        features = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return self.model.get_text_features(**features).to(self.device)

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = torch.from_numpy(audio).to(self.device, dtype=torch.float32)
        return self.feature_extractor(audio)

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

    fake_audios = np.random.normal(0, 1, (10, 10000))
    similarity = clap.compute_similarity(fake_audios, targets)

    print(similarity.shape)  # (5, 10)
    print(similarity)


if __name__ == "__main__":
    test_clap()
