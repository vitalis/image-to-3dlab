"""The leg-reach check compares world height to the world marker, not local Z."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_parented_vertex_uses_world_z():
    reach = _load("leg_reach")
    # Local Z is 0.2. The mesh sits 10 units up in the world. The head marker is at 1.5.
    # Comparing local Z to the marker says the leg is fine. World Z says it reaches.
    assert reach.world_z_of((0.0, 0.0, 0.2), (0.0, 0.0, 1.0, 10.0)) == pytest.approx(
        10.2
    )
    span = reach.height_span([10.0, 11.0])
    assert reach.reaches_head(10.2, head_z=1.5, span=span) is True
    assert reach.reaches_head(0.2, head_z=1.5, span=1.0) is False


def test_a_clean_world_comparison_stays_clean():
    reach = _load("leg_reach")
    assert reach.reaches_head(0.4, head_z=1.0, span=1.0) is False
    assert reach.bind_exit("CLEAN") == 0
    assert reach.bind_exit("EMPTY GROUPS") == 0
    assert reach.bind_exit("ENVELOPE FALLBACK") == 1
    assert reach.bind_exit("ENVELOPE-LIKE BLEED") == 1


def test_the_rig_program_calls_the_world_check():
    rig = _load("blender_build_rig")
    code = rig.build_code(0.09, 0.15)
    assert "reaches_head(" in code
    assert "matrix_world @ v.co" in code
    assert "v.co.z" not in code
    assert "bind_exit" in inspect.getsource(rig.main)


def test_the_voxel_program_calls_the_same_check():
    weights = _load("blender_voxel_weights_code")
    code = weights.build_code(0.012, 0.15, False)
    assert "reaches_head(" in code
    assert "matrix_world @ v.co" in code
    assert "v.co.z" not in code
    caller = _load("blender_voxel_weights")
    assert "bind_exit" in inspect.getsource(caller.main)
