"""A normal bake that writes nothing must fail, and every mesh in the file is in it."""

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


def test_a_missing_png_or_null_write_is_a_failure():
    bake = _load("blender_bake_normals")
    assert bake.normal_bake_exit(None, False) == 1
    assert bake.normal_bake_exit(None, True) == 1
    assert bake.normal_bake_exit("/tmp/out.png", False) == 1
    assert bake.normal_bake_exit("/tmp/out.png", True) == 0


def test_more_than_one_mesh_is_joined_and_none_is_refused():
    bake = _load("blender_bake_normals")
    assert bake.meshes_to_bake(1) == "single"
    assert bake.meshes_to_bake(3) == "join"
    with pytest.raises(ValueError):
        bake.meshes_to_bake(0)


def test_the_bake_programs_call_those_decisions():
    bake = _load("blender_bake_normals")
    retopo = _load("blender_retopo_bake")
    reunwrap = _load("blender_reunwrap_bake")
    assert "meshes_to_bake(" in bake.TEMPLATE
    assert "normal_bake_exit" in inspect.getsource(bake.main)
    assert "meshes_to_bake" in inspect.getsource(retopo.main)
    assert "meshes_to_bake" in inspect.getsource(reunwrap.main)
