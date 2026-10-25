"""Stage-bake waits for the frames it asked for, and says so when the reply is empty."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_stage_bake.py"


def _load():
    spec = importlib.util.spec_from_file_location("stage_bake", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timeout_grows_with_the_frame_count_and_unknown_ranges_wait():
    mod = _load()
    assert mod.read_timeout(1) >= 120
    assert mod.read_timeout(10) >= 300
    assert mod.read_timeout(None) > mod.read_timeout(1)
    assert mod.frame_count(1, 10, 1) == 10
    assert mod.frame_count(None, None, 1) is None


def test_a_reply_without_the_bake_line_fails():
    mod = _load()
    assert mod.stage_bake_exit("still working") == 1
    assert mod.stage_bake_exit('STAGE_BAKE {"frame_count": 4}') == 0
    assert "read_timeout" in inspect.getsource(mod.main)
    assert "stage_bake_exit" in inspect.getsource(mod.main)
    assert "settimeout(120)" not in inspect.getsource(mod.send)
