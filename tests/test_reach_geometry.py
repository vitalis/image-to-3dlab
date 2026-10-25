"""Chain length and ankle offset are measured in one space, on the facing axis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_quadruped_walk.py"


def _load():
    spec = importlib.util.spec_from_file_location("quadruped_walk_reach", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y, self.z - other.z)

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]

    @property
    def length(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5


class Scale:
    def __init__(self, scale):
        self.scale = scale

    def __matmul__(self, vector):
        return Vec(vector.x * self.scale, vector.y * self.scale, vector.z * self.scale)


class Bone:
    def __init__(self, head, tail=None):
        self.head = head
        self.head_local = head
        self.tail_local = head if tail is None else tail


class ArmatureData:
    def __init__(self, bones):
        self.bones = bones


class Pose:
    def __init__(self, bones):
        self.bones = bones


class Rig:
    def __init__(self, bones, scale, head, hips):
        self.data = ArmatureData(bones)
        self.matrix_world = Scale(scale)
        self.pose = Pose(
            {
                "head": Bone(Vec(*head)),
                "hips": Bone(Vec(*hips)),
            }
        )


def _rig(scale):
    bones = {
        "ORG-front_thigh.L": Bone(Vec(0, 0, 1), Vec(0, 0, 0)),
        "ORG-front_shin.L": Bone(Vec(0, 0, 0), Vec(1, 0, 0)),
    }
    # Head is further from the hips on X than on Y, so forward is +X.
    return Rig(bones, scale, head=(5, 0.2, 0), hips=(0, 0, 0))


def test_reach_geometry_uses_world_length_and_the_facing_axis():
    mod = _load()
    got = mod.reach_geometry(_rig(2), None, "front_thigh.L", "front_shin.L")
    # Each rest bone is length 1. Scale 2 makes the world chain 4, not 2.
    # The ankle sits 2 on X and 0 on Y. Forward is X, so the stance is 2, not 0.
    assert got["chain_length"] == pytest.approx(4.0)
    assert got["vertical_drop"] == pytest.approx(2.0)
    assert got["horizontal_at_rest"] == pytest.approx(2.0)


def test_a_metarig_does_not_replace_the_rig_measurement():
    mod = _load()
    metarig_bones = {
        "front_thigh.L": Bone(Vec(0, 0, 0), Vec(10, 0, 0)),
        "front_shin.L": Bone(Vec(0, 0, 0), Vec(0, 10, 0)),
    }
    metarig = Rig(metarig_bones, 1, head=(0, 1, 0), hips=(0, 0, 0))
    got = mod.reach_geometry(_rig(2), metarig, "front_thigh.L", "front_shin.L")
    assert got["chain_length"] == pytest.approx(4.0)
    assert got["horizontal_at_rest"] == pytest.approx(2.0)
