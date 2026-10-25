"""Wind has to have something rigid to bend around."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_wind_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("wind_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_rigid_vertex_is_an_error():
    mod = _load()
    with pytest.raises(mod.NoRigidVertex):
        mod.require_rigid_anchors([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [True, True])


def test_rigid_vertices_are_the_anchors():
    mod = _load()
    anchors = mod.require_rigid_anchors(
        [(0.0, 1.0, 2.0), (3.0, 0.0, 0.0)], [False, True]
    )
    assert anchors == [(0.0, 1.0, 2.0)]
