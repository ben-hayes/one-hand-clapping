"""
VST Host implementation using DawDreamer.
"""
from pathlib import Path
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import dawdreamer as daw
import numpy as np
import ray

from ohc.vst.vst_base import VSTBase


class VSTHostDawDreamer(VSTBase):
    """
    VST Host implementation using DawDreamer.
    """

    def __init__(
        self,
        vst_path: str,
        inactive_param_behaviour: str,
        sample_rate: int,
        block_size: int,
        num_cpus: Optional[int] = None,
    ) -> None:
        """
        Constructor for the VSTHostDawDreamer class.
        """
        super().__init__(
            vst_path, inactive_param_behaviour, sample_rate, block_size=block_size
        )

        # Initialise ray if not already initialised
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, log_to_driver=False)

    def _initiate_synth(
        self,
        vst_path: Union[str, Path],
        inactive_param_behaviour: Literal["random", "fixed"],
        sample_rate: int,
        block_size: int,
    ) -> None:
        """
        Initialises the VST plugin using DawDreamer.
        """
        self.vst_path = vst_path
        self.inactive_param_behaviour = inactive_param_behaviour
        self.sample_rate = sample_rate
        self.block_size = block_size

        # Initialise the render engine
        self.engine = daw.RenderEngine(self.sample_rate, self.block_size)
        self.synth = self.engine.make_plugin_processor("synth", self.vst_path)

        # Map from parameter names to parameter indices
        self.params = self.synth.get_parameters_description()
        self.param_map = {p["name"]: p["index"] for p in self.params}
        self.active_params = []
        self.active_indices = []

    def list_params(
        self,
        filter_midi_cc: Optional[bool] = True,
    ) -> List[str]:
        """
        Lists the parameter names of the VST plugin.

        Optionally filters out MIDI CC parameters which dawdreamer returns by default,
        and there are ALOT of them.
        """
        param_names = [p["name"] for p in self.params]
        if filter_midi_cc:
            param_names = [p for p in param_names if not p.startswith("MIDI CC")]
        return param_names

    def set_active_params(self, active_params: List[str]) -> None:
        """
        Sets the active parameters for rendering of the VST plugin.
        """
        self.active_params = active_params
        self.active_indices = [self.param_map[p] for p in active_params]

    def render(
        self,
        params: np.ndarray,  # A batch of parameter settings (batch_size, num_params)
        midi_note: int,  # Midi note number to play
        note_duration_in_seconds: float,  # Duration of the note in seconds
        tail_duration_in_seconds: float,  # Duration of the tail to render after note
    ) -> None:
        """
        Abstract method that must be implemented by the subclass.
        Renders the VST plugin.

        Receives a batch of parameter settings and should render audio for each setting
        as a background process. The callback function should be called for each
        rendered audio signal along with the index of the parameter setting associated
        with that rendered audio signal.
        Importantly, this is a non-blocking function. It should return immediately after
        dispatching the background rendering process.
        """

        assert len(params.shape) == 2, "params must be (batch_size, num_params)"

        if len(self.active_indices) == 0:
            raise ValueError("No active parameters set. Use set_active_params()")

        if params.shape[1] != len(self.active_indices):
            raise ValueError(
                "Number of parameters must match the active parameters "
                "set by set_active_params()"
            )

        # Create a list of parameter settings keyed on their synthesizer preset
        # index and the index they are passed within the batch
        synth_params = []
        for batch_idx, p in enumerate(params):
            param_dict = {}
            for idx, val in zip(self.active_indices, p):
                param_dict[idx] = np.clip(val, 0.0, 1.0)
            synth_params.append((param_dict, batch_idx))

        synth_args = {
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "vst_path": self.vst_path,
            "midi_note": midi_note,
            "velocity": 127,  # Default velocity of 127 \m/ \m/
            "note_duration_in_seconds": note_duration_in_seconds,
            "tail_duration_in_seconds": tail_duration_in_seconds,
        }

        # Render the audio for each parameter setting using ray remote
        remotes = [_background_render.remote(*p, **synth_args) for p in synth_params]
        return remotes

    def render_now(
        self,
        params: np.ndarray,  # A batch of parameter settings
        midi_note: int,  # Midi note number to play
        note_duration_in_seconds: float,  # Duration of the note in seconds
        tail_duration_in_seconds: float,  # Duration of the tail to render after note
    ) -> List[np.ndarray]:
        """
        Renders the VST plugin.

        Receives a batch of parameter settings and renders audio for each setting
        as a blocking process. Returns the rendered audio signals.
        """

        remotes = self.render(
            params,
            midi_note,
            note_duration_in_seconds,
            tail_duration_in_seconds,
        )
        return ray.get(remotes)


@ray.remote
def _background_render(
    params: Dict[int, float],
    batch_idx: int,
    sample_rate: int,
    block_size: int,
    vst_path: str,
    midi_note: int,
    velocity: int,
    note_duration_in_seconds: float,
    tail_duration_in_seconds: float,
):
    """
    Background rendering function for the VST plugin.
    """
    # Initialise the render engine and synth
    engine = daw.RenderEngine(sample_rate, block_size)
    synth = engine.make_plugin_processor("synth", vst_path)

    engine.load_graph(
        [
            (synth, []),
        ]
    )

    synth.add_midi_note(midi_note, velocity, 0.0, note_duration_in_seconds)

    # Set the parameter values
    for idx, value in params.items():
        synth.set_parameter(idx, value)

    # Render the audio
    engine.render(note_duration_in_seconds + tail_duration_in_seconds)
    audio = engine.get_audio()

    return audio, batch_idx
