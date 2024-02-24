from dataclasses import dataclass
from typing import List, Optional

from .vst.vst_base import VSTBase
from .vst.vst_dawdreamer import VSTHostDawDreamer
from .fitness.clap import CLAPSimilarity
from .fitness.fitness import FitnessFunction


@dataclass
class SearcherConfig:
    # VSTi Host Parameters
    vst_path: str
    vst_preset: Optional[str] = None
    num_cpus: int = 8

    # Rendering Parameters
    midi_note: int = 50
    note_duration_in_seconds: float = 1.0
    tail_duration_in_seconds: float = 0.5

    # CLAP Parameters
    clap_model: str = "laion/larger_clap_general"
    clap_batch_size: int = 32

    # Evolution Parameters
    population_size: int = 100

    # Search Parameters
    text_targets: Optional[List[str]] = None
    audio_targets: Optional[List[str]] = None


class Searcher:
    initialised: bool = False

    _config: SearcherConfig

    vsti_host: VSTBase
    clap_similarity: CLAPSimilarity
    evolutionary_algorithm: "EvoCLAP"
    fitness_function: FitnessFunction

    def __init__(self, config: SearcherConfig):
        self.config = config

        self._init_vsti_host(config)
        self._init_clap_similarity(config)
        self._init_fitness_function(config, self.clap_similarity)

    def _init_vsti_host(self, config: SearcherConfig):
        self.vsti_host = VSTHostDawDreamer(
            vst_path=config.vst_path,
            inactive_param_behaviour="fixed",
            sample_rate=48000,
            block_size=512,
            num_cpus=config.num_cpus,
        )

    def _init_clap_similarity(self, config: SearcherConfig):
        self.clap_similarity = CLAPSimilarity(
            model_name=config.clap_model,
            sample_rate=48000,
        )

    def _init_fitness_function(
        self, config: SearcherConfig
    ):
        self.fitness_function = FitnessFunction(
            vsti_host=self.vsti_host,
            clap_similarity=self.clap_similarity,
            text_targets=config.text_targets,
            audio_targets=config.audio_targets,
            clap_batch_size=config.clap_batch_size,
            midi_note=config.midi_note,
            note_duration_in_seconds=config.note_duration_in_seconds,
            tail_duration_in_seconds=config.tail_duration_in_seconds,
        )

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: SearcherConfig):
        if (
            config.vst_path != self.config.vst_path
            or config.num_cpus != self.config.num_cpus
            or config.vst_preset != self.config.vst_preset
        ):
            self._init_vsti_host(config)

        if config.clap_model != self.config.clap_model:
            self._init_clap_similarity(config)

        self._init_fitness_function(config, self.clap_similarity)

        self._config = config
