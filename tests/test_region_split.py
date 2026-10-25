"""Tests for splitting a mesh into per-region material slots.

The point of the split is that 2048 is the cap *per material*: three regions means three
atlases, so the budget triples without changing any generation setting. Cuts are fractions
of whatever mesh is in front of the script, not one creature's coordinates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_split_regions.py"
WALK = Path(__file__).resolve().parents[1] / "scripts" / "blender_quadruped_walk.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load(SCRIPT, "split_regions")


@pytest.fixture(scope="module")
def walk():
    return _load(WALK, "quadruped_walk_axis")


def test_faces_land_in_the_expected_region(mod):
    """0 is the tail end of the bounds, 1 is the head end. The cuts are fractions."""
    got = mod.classify_faces([0.0, 50.0, 100.0], head_cut=2 / 3, tail_cut=1 / 3)
    assert got == ["tail", "body", "head"]


def test_a_shifted_mesh_uses_its_own_bounds(mod):
    """The same fractions on a mesh that does not sit at the origin."""
    got = mod.classify_faces([100.0, 150.0, 200.0], head_cut=2 / 3, tail_cut=1 / 3)
    assert got == ["tail", "body", "head"]


def test_boundaries_are_exclusive_and_fall_to_body(mod):
    """A face exactly on a cut must not be double-claimed; body is the safe default."""
    got = mod.classify_faces([0.0, 10.0, 20.0], head_cut=0.5, tail_cut=0.0)
    assert got == ["body", "body", "head"]


def test_every_face_gets_exactly_one_region(mod):
    positions = [i * 3.0 for i in range(11)]
    got = mod.classify_faces(positions, head_cut=2 / 3, tail_cut=1 / 3)
    assert len(got) == len(positions)
    assert set(got) <= set(mod.REGION_NAMES)
    assert sum(mod.region_counts(got).values()) == len(positions)


def test_inverted_cuts_are_rejected(mod):
    with pytest.raises(ValueError):
        mod.classify_faces([0.0, 1.0], head_cut=0.2, tail_cut=0.8)


def test_cuts_outside_the_bounds_are_rejected(mod):
    with pytest.raises(ValueError):
        mod.classify_faces([0.0, 1.0], head_cut=1.5, tail_cut=0.2)


def test_a_flat_mesh_is_refused(mod):
    with pytest.raises(ValueError):
        mod.classify_faces([4.0, 4.0], head_cut=2 / 3, tail_cut=1 / 3)


def test_an_empty_region_is_refused(mod):
    assignments = mod.classify_faces([0.0, 1.0], head_cut=0.5, tail_cut=0.0)
    with pytest.raises(ValueError):
        mod.require_every_region(assignments)


def test_pack_faces_are_only_the_named_region(mod):
    assignments = ["head", "body", "head", "tail"]
    assert mod.pack_face_indices(assignments, "head") == [0, 2]
    assert mod.pack_face_indices(assignments, "tail") == [3]


def test_defaults_are_fractions_not_one_creature(mod):
    args = mod.build_parser().parse_args(["asset.glb", "out"])
    assert 0.0 <= args.tail_cut < args.head_cut <= 1.0
    assert (args.head_cut, args.tail_cut) != (0.25, -0.10)


def test_the_pack_program_calls_the_classifier(mod, tmp_path):
    code = mod.region_program(
        tmp_path / "asset.glb", tmp_path / "out", 2048, 2 / 3, 1 / 3
    )
    assert "classify_faces(" in code
    assert "pack_face_indices(" in code
    assert "require_every_region(" in code
    assert "forward_axis(" in code
    assert "uv.select_all" not in code
    assert "moss_fox" not in code
    assert "-centre.y" not in code
    assert code.index("classify_faces(") < code.index("pack_islands")


def test_front_axis_with_no_rig_is_the_shared_fallback(walk):
    """The same rule the rig tools use when the head and hips are missing: -Y."""
    assert walk.forward_axis(None) == (1, -1.0)


def test_density_triples_when_one_atlas_becomes_three(mod):
    """The arithmetic the whole exercise rests on."""
    total = 101_298
    before = mod.texel_density(total, 2048)
    thirds = [mod.texel_density(total // 3, 2048) for _ in range(3)]
    assert before == pytest.approx(41.4, abs=0.5)
    for d in thirds:
        assert d / before == pytest.approx(3.0, abs=0.05)


def test_density_matches_the_measured_split(mod):
    """Pinned to the real face counts, so a change in the cuts is visible."""
    assert mod.texel_density(35_824, 2048) == pytest.approx(117.0, abs=1.0)
    assert mod.texel_density(28_647, 2048) == pytest.approx(146.4, abs=1.0)


def test_empty_region_does_not_divide_by_zero(mod):
    assert mod.texel_density(0, 2048) == 0.0
