"""
Classes for hosting VST plugins.
"""
from pathlib import Path
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np


class VSTBase:
    """
    Generic Base Class for hosting VST plugins.
    """

    def __init__(
        self,
        vst_path: Union[str, Path],  # Path to the VST plugin
        inactive_param_behaviour: Literal["random", "fixed"],
    ) -> None:
        """
        Constructor for the VSTBase class.
        """
        self._initiate_synth(vst_path, inactive_param_behaviour)

    def _initiate_synth(
        self,
        vst_path: Union[str, Path],
        inactive_param_behaviour: Literal["random", "fixed"],
    ) -> None:
        """
        Abstract method that must be implemented by the subclass.
        Handles any initialisation of the VST plugin.
        """
        raise NotImplementedError

    def list_params(self) -> List[str]:
        """
        Abstract method that must be implemented by the subclass.
        Lists the parameters of the VST plugin.
        """
        raise NotImplementedError

    def set_inactive_params(
        self,
        inactive_params: List[str],  # Names of parameters that should be inactive
        param_values: Optional[List[float]],  # Optional values for inactive params
    ) -> None:
        """
        Abstract method that must be implemented by the subclass.
        Sets the inactive parameters of the VST plugin.
        """
        raise NotImplementedError

    def render(
        self,
        params: np.ndarray,  # A batch of parameter settings
        midi_note: int,  # Midi note number to play
        note_duration_in_seconds: float,  # Duration of the note in seconds
        tail_duration_in_seconds: float,  # Duration of the tail to render after note
        callback: Callable[[np.ndarray, int], None],  # Callback
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
        raise NotImplementedError
