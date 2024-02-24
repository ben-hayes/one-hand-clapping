"""
Unit tests for VST hosting
"""
import os

from dotenv import load_dotenv

from ohc.vst import VSTHost

load_dotenv()
PLUGIN_PATH = os.getenv("TEST_PLUGIN")


def test_vst_host_dawdreamer():
    VSTHost(
        vst_path=PLUGIN_PATH,
        inactive_param_behaviour="random",
        sample_rate=48000,
        block_size=512,
    )
