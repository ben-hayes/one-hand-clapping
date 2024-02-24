"""
Unit tests for VST hosting
"""
from ohc.vst import VSTHost


def test_vst_host_dawdreamer():
    vst_file = "test/data/test_plugin.vst"
    VSTHost(
        vst_path=vst_file,
        inactive_param_behaviour="random",
        sample_rate=48000,
        block_size=512,
    )
