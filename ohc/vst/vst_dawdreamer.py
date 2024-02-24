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
    ) -> None:
        """
        Constructor for the VSTHostDawDreamer class.
        """
        super().__init__(
            vst_path, inactive_param_behaviour, sample_rate, block_size=block_size
        )

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

    def list_params(
        self,
        filter_midicc: Optional[
            bool
        ] = True,  # Whether to filter out MIDI CC parameters
    ) -> List[str]:
        """
        Lists the parameter names of the VST plugin.
        """
        params = self.synth.get_parameters_description()
        if filter_midicc:
            params = [p for p in params if not p["name"].startswith("MIDI")]

        param_names = [p["name"] for p in params]
        return param_names

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
        Background rendering function
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

        # Merge the parameter settings with the inactive parameter settings
        synth_params = []
        for i, p in enumerate(params):
            param_dict = {j: p[j] for j in range(len(p))}
            synth_params.append((param_dict, i))

        synth_args = {
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "vst_path": self.vst_path,
            "midi_note": midi_note,
            "velocity": 127,  # Default velocity of 127
            "note_duration_in_seconds": note_duration_in_seconds,
            "tail_duration_in_seconds": tail_duration_in_seconds,
        }

        # Render the audio for each parameter setting using ray remote
        remotes = [
            self._background_render.remote(*p, **synth_args) for p in synth_params
        ]
        return remotes
