"""
VST Host implementation using DawDreamer.
"""
import dawdreamer as daw  # noqa

from ohc.vst.vst_base import VSTBase


class VSTHostDawDreamer(VSTBase):
    """
    VST Host implementation using DawDreamer.
    """

    def __init__(self, vst_path: str, inactive_param_behaviour: str) -> None:
        """
        Constructor for the VSTHostDawDreamer class.
        """
        super().__init__(vst_path, inactive_param_behaviour)
