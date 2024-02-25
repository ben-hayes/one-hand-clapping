from queue import Queue
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import evotorch
import numpy as np
import ray
import torch
from evotorch.decorators import vectorized

from ..vst import VSTHost
from .clap import CLAPSimilarity


class FitnessFunction:
    vsti_host: VSTHost
    clap_similarity: CLAPSimilarity
    text_target: Optional[List[str]] = None
    audio_targets: Optional[torch.Tensor] = None

    text_target_embeddings: Optional[torch.Tensor] = None
    audio_target_embeddings: Optional[torch.Tensor] = None

    def __init__(
        self,
        vsti_host: VSTHost,
        clap_similarity: CLAPSimilarity,
        text_target: Optional[List[str]] = None,
        audio_targets: Optional[torch.Tensor] = None,
        clap_batch_size: int = 32,
        midi_note: int = 48,
        note_duration_in_seconds: float = 1.0,
        tail_duration_in_seconds: float = 0.5,
        queue_timeout: float = 10.0,
    ):
        self.vsti_host = vsti_host
        self.clap_similarity = clap_similarity

        self.text_target = text_target
        self.audio_targets = audio_targets

        if text_target is None and audio_targets is None:
            raise ValueError(
                "At least one of text_target or audio_targets must be provided"
            )

        self.clap_batch_size = clap_batch_size
        self.midi_note = midi_note
        self.note_duration_in_seconds = note_duration_in_seconds
        self.tail_duration_in_seconds = tail_duration_in_seconds

        self.queue_timeout = queue_timeout

    @property
    def target_embeddings(self) -> torch.Tensor:
        if self.text_target_embeddings is None and self.text_target is not None:
            self.text_target_embeddings = self.clap_similarity.get_text_embedding(
                self.text_target
            )

        if self.audio_target_embeddings is None and self.audio_targets is not None:
            self.audio_target_embeddings = self.clap_similarity.get_audio_embedding(
                self.audio_targets
            )

        embeddings = []

        if self.text_target_embeddings is not None:
            embeddings.append(self.text_target_embeddings)

        if self.audio_target_embeddings is not None:
            embeddings.append(self.audio_target_embeddings)

        return torch.cat(embeddings, dim=0)

    def _audio_ready_callback(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        index_in_batch: int,
        audio_queue: Queue,
    ) -> None:
        audio_queue.put((index_in_batch, audio))

    def _process_clap_batch(
        self, clap_batch: List[Tuple[int, np.ndarray]]
    ) -> torch.Tensor:
        print(f"Processing batch of {len(clap_batch)} audio clips")

        audios, indices = zip(*clap_batch)
        indices = torch.tensor(indices, dtype=torch.long)
        audios = np.stack(audios, axis=0)

        return indices, self.clap_similarity.compute_similarity(
            audios, self.target_embeddings
        )

    def _consume_audio(self, ray_objects: List[ray.ObjectRef], items_total: int):
        items_processed = 0

        clap_batch = []
        outputs = []

        while items_processed < items_total:
            ready_objects, ray_objects = ray.wait(ray_objects, num_returns=1)

            for ready_object in ready_objects:
                audio, idx = ray.get(ready_object)
                audio = np.mean(audio, axis=0)
                clap_batch.append((audio, idx))

                items_remaining = items_total - items_processed
                clap_threshold = (
                    self.clap_batch_size
                    if items_remaining > self.clap_batch_size
                    else items_remaining
                )

                if len(clap_batch) >= clap_threshold:
                    outputs.append(self._process_clap_batch(clap_batch))
                    items_processed += len(clap_batch)

                    clap_batch = []

        if len(clap_batch) > 0:
            raise RuntimeError(
                f"{len(clap_batch)} items were not processed."
                "We should *not* reach this point."
            )
        return outputs

    @vectorized
    def compute(self, batch: evotorch.SolutionBatch) -> torch.Tensor:
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        elif batch.ndim != 2:
            raise ValueError("Batch must be 2D tensor")

        params = batch.detach().cpu().numpy()
        ray_waitables = self.vsti_host.render(
            params,
            self.midi_note,
            self.note_duration_in_seconds,
            self.tail_duration_in_seconds,
        )

        outputs = self._consume_audio(ray_waitables, len(batch))

        indices, similarities = zip(*outputs)
        indices = torch.cat(indices, dim=0)
        similarities = torch.cat(similarities, dim=0)
        similarities = similarities[indices.argsort()]
        # similarities = torch.split(similarities, 1, dim=1)
        # similarities = (s.squeeze(1) for s in similarities)
        return similarities


if __name__ == "__main__":
    import random
    import time

    class FakeVstiHost:
        def render(self, params, midi_note, note_duration, tail_duration):
            @ray.remote
            def render_thread(i):
                sleep_time = random.uniform(0.1, 0.5)
                time.sleep(sleep_time)
                print(f"Callback {i} after {sleep_time} seconds")
                return np.random.uniform(0, 1, (2, 3)), i

            return [render_thread.remote(i) for i in range(params.shape[0])]

    class FakeCLAPSimilarity:
        device = torch.device("cpu")

        def get_text_embedding(self, text: List[str]) -> torch.Tensor:
            return torch.rand(len(text), 3)

        def get_audio_embedding(self, audio: torch.Tensor) -> torch.Tensor:
            sleep_time = random.uniform(0.01, 0.1)
            time.sleep(sleep_time)
            print(f"CLAP audio after {sleep_time} seconds")
            return torch.rand(len(audio), 3)

        def compute_similarity(
            self, audio: torch.Tensor, target_embeddings: torch.Tensor
        ) -> torch.Tensor:
            sleep_time = random.uniform(0.1, 2.5)
            time.sleep(sleep_time)
            print(f"CLAP similarity after {sleep_time} seconds")
            return torch.rand(len(audio), len(target_embeddings))

    class FakeSolutionBatch:
        def __init__(self, values: torch.Tensor):
            self.values = values

        def __len__(self):
            return len(self.values)

    fitness = FitnessFunction(
        FakeVstiHost(),
        FakeCLAPSimilarity(),
        text_target=["hello", "world"],
        clap_batch_size=5,
    )

    batch = FakeSolutionBatch(
        torch.rand(50, 3),
    )

    fitness.compute(batch)
