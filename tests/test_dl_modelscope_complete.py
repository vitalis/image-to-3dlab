"""A short ModelScope file is not a finished download."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "hunyuan_mlx"
    / "shape"
    / "scripts"
    / "dl_modelscope.py"
)
spec = importlib.util.spec_from_file_location("dl_modelscope_complete", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_a_short_file_is_not_complete(tmp_path):
    out = tmp_path / "partial.safetensors"
    out.write_bytes(b"x" * 100)
    assert mod.is_complete(str(out), 1_000_000) is False
