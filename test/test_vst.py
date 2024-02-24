"""
Unit tests for VST hosting
"""
import os

from dotenv import load_dotenv

from ohc.vst import VSTHost

load_dotenv()
PLUGIN_PATH = os.getenv("TEST_PLUGIN")


def test_vst_host_dawdreamer_initialise():
    VSTHost(
        vst_path=PLUGIN_PATH,
        inactive_param_behaviour="random",
        sample_rate=48000,
        block_size=512,
    )


def test_vst_list_params():
    vst = VSTHost(
        vst_path=PLUGIN_PATH,
        inactive_param_behaviour="random",
        sample_rate=48000,
        block_size=512,
    )
    params = vst.list_params()
    assert len(params) > 0
    assert isinstance(params[0], str)
