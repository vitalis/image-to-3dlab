"""A multi-part asset is scaled once, from the combined height."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_lineup_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("lineup_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parts_share_the_scale_of_the_combined_height():
    mod = _load()
    # Body spans 0..2. An eye on the head spans 1.8..2.0. One height, one scale.
    # Scaling each part from its own height would make the eye five times too big.
    assert mod.shared_scale([(0.0, 2.0), (1.8, 2.0)]) == pytest.approx(0.5)
    assert mod.shared_scale([(0.0, 0.0)]) == 1.0


def test_import_asset_uses_that_scale():
    mod = _load()
    source = inspect.getsource(mod.import_asset)
    assert "shared_scale" in source
    assert "dimensions.z" not in source
