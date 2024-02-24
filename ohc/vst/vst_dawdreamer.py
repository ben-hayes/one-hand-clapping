"""
VST Host implementation using DawDreamer.
"""
from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import dawdreamer as daw

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
