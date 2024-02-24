"""
Unit tests for VST hosting
"""
import os

import numpy as np
import pytest
from dotenv import load_dotenv

from ohc.vst import VSTHost

load_dotenv()
PLUGIN_PATH = os.getenv("TEST_PLUGIN")


@pytest.fixture
def vst():
    vst = VSTHost(
        vst_path=PLUGIN_PATH,
        inactive_param_behaviour="random",
        sample_rate=48000,
        block_size=512,
    )
    return vst


def test_vst_host_dawdreamer_initialise(vst):
    assert vst.vst_path == PLUGIN_PATH
    assert vst.inactive_param_behaviour == "random"
    assert vst.sample_rate == 48000
    assert vst.block_size == 512


def test_vst_list_params(vst):
    params = vst.list_params()
    assert len(params) > 0
    assert isinstance(params[0], str)


def test_vst_render(vst):
    params = np.random.rand(16, len(vst.list_params()))
    vst.render(params, 48, 1.0, 1.0)
