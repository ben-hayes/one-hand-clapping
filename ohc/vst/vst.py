"""
Classes for hosting VST plugins.
"""

from pathlib import Path
from typing import Union

class VSTBase:
    """
    Generic Base Class for hosting VST plugins.
    """

    def __init__(
        self,
        vst_path: Union[str, Path], # Path to the VST plugin
    ) -> None:
        pass