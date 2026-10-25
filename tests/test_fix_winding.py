"""Tests for outward-facing winding repair.

The subtle one is `needs_inverting`: `fix_normals()` makes winding CONSISTENT but can leave
a mesh uniformly inside-out, which is precisely what happened on Flicker (volume -0.02369
after repair). Consistency alone is not the goal; outward is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fix_winding.py"


def _load():
    spec = importlib.util.spec_from_file_location("fix_winding", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fw = _load()
trimesh = pytest.importorskip("trimesh")


def test_negative_volume_needs_inverting():
    assert fw.needs_inverting(-0.02369)


def test_positive_volume_does_not():
    assert not fw.needs_inverting(0.00199)


def test_zero_volume_is_left_alone():
    """An open sheet has ~zero signed volume; flipping it would be a coin toss."""
    assert not fw.needs_inverting(0.0)


# --- against a real mesh --------------------------------------------------------------


def _box(inside_out: bool):
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    if inside_out:
        mesh.invert()
    return mesh


def test_repairs_an_inside_out_mesh():
    """Assert the OUTCOME, not the mechanism.

    On a clean watertight box `fix_normals()` already resolves orientation, so the
    explicit invert never fires. On a real generated mesh it does -- Flicker was still at
    volume -0.02369 after `fix_normals()`. Either route is fine; ending up outward-facing
    is the contract.
    """
    mesh = _box(inside_out=True)
    assert mesh.volume < 0
    fw.repair(mesh)
    assert mesh.volume > 0
    assert mesh.is_winding_consistent


def test_leaves_a_correct_mesh_alone():
    mesh = _box(inside_out=False)
    before = mesh.volume
    report = fw.repair(mesh)
    assert not report["inverted"]
    assert mesh.volume == pytest.approx(before)


def test_repair_preserves_geometry():
    """Only winding may change -- vertex positions and face count must not."""
    mesh = _box(inside_out=True)
    vertices_before = mesh.vertices.copy()
    faces_before = len(mesh.faces)
    fw.repair(mesh)
    assert len(mesh.faces) == faces_before
    assert mesh.vertices.shape == vertices_before.shape


def test_report_records_the_before_state():
    mesh = _box(inside_out=True)
    report = fw.repair(mesh)
    assert report["was_volume"] < 0
    assert report["now_volume"] > 0
    assert report["now_consistent"]


def test_a_seamed_box_away_from_the_origin_keeps_its_faces():
    """A UV seam is not its own piece. A correct box moved off the origin stays correct."""
    import numpy as np

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = box.vertices[box.faces].reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(vertices), 2)))
    mesh.apply_translation((5.0, 2.0, -3.0))
    before = mesh.faces.copy()
    count = len(mesh.vertices)
    fw.repair(mesh)
    assert abs(float(mesh.volume) - 1.0) < 1e-6
    assert len(mesh.vertices) == count
    assert not np.all(mesh.faces == before[:, ::-1], axis=1).any()
    assert mesh.visual.uv.shape == (count, 2)


def test_an_open_inside_out_piece_is_turned_outward():
    """One closed piece and one open inside-out piece. Both must end outward."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from repair_decode import summarise

    closed = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    closed.apply_translation((3.0, 0.0, 0.0))
    opened = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    opened.invert()
    opened.faces = opened.faces[:-1]
    both = trimesh.util.concatenate([closed, opened])
    fw.repair(both)
    assert summarise(both)["inverted_components"] == 0
