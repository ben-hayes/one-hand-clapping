import os

from dotenv import load_dotenv

load_dotenv()

# Don't run the VST tests if the VST plugin is not available
collect_ignore = []
if os.getenv("TEST_PLUGIN", None) is None:
    collect_ignore.append("test_vst.py")
