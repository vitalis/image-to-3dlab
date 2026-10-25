"""Paw planting and gait phases live outside the Blender string."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nearest_vertex_uses_the_space_it_is_given():
    pose = _load("walk_pose")
    # Index 0 is near in some other space. Index 1 is near the marker.
    points = [(10.0, 0.0, 0.0), (0.1, -0.2, 0.3)]
    assert pose.nearest_vertex(points, (0.0, 0.0, 0.0)) == 1


def test_plant_shift_is_the_gap_from_the_floor():
    pose = _load("walk_pose")
    assert pose.plant_shift(0.1, [0.4, 0.5]) == pytest.approx(-0.3)


def test_one_cycle_returns_to_the_same_leg_angles():
    pose = _load("walk_pose")
    phases = pose.PHASES["trot"]
    kwargs = dict(
        swing_front=24.0,
        swing_back=20.0,
        bend_front=34.0,
        bend_back=38.0,
        wrist=40.0,
        ankle=22.0,
        base_bend=8.0,
        front_crouch=0.0,
        crouch=0.0,
        front_sign=-1.0,
        toe=20.0,
        shoulder_tuck=0.0,
        chest_drop=0.0,
        hind_lead_l=0.0,
        hind_lead_r=0.0,
    )
    start = pose.leg_pose(0.0, "frontL", phases["frontL"], **kwargs)
    end = pose.leg_pose(1.0, "frontL", phases["frontL"], **kwargs)
    assert start.keys() == end.keys()
    for name in start:
        assert start[name] == pytest.approx(end[name])
    assert pose.body_location_z(0.0, 0.0, 0.0, 0.0, 0.016) == pytest.approx(
        pose.body_location_z(1.0, 0.0, 0.0, 0.0, 0.016)
    )


def test_the_walk_program_calls_the_module_in_one_space():
    walk = _load("blender_walk_cycle")
    args = walk.build_parser().parse_args([])
    assert args.head_yaw == 0.0
    code = walk.code_for(args)
    assert "nearest_vertex" in code
    assert "leg_pose" in code
    assert "body_pose" in code
    assert "matrix_world @ v.co" in code
    assert code.index("matrix_world @ v.co") < code.index("nearest_vertex")
    assert "v.co.copy()" not in code
    assert "** 0.7" not in code
    assert "-32" not in code
    turned = walk.code_for(walk.build_parser().parse_args(["--head-yaw", "-15"]))
    assert "-15" in turned


def test_idle_angles_default_to_zero_and_an_explicit_angle_is_kept():
    idle = _load("blender_idle_cycle")
    parser = idle.build_parser()
    assert parser.parse_args([]).head_yaw == 0.0
    assert parser.parse_args([]).ear_turn == 0.0
    straight = idle.code_for(parser.parse_args([]))
    assert "-32" not in straight
    assert "-62" not in straight
    asked = parser.parse_args(["--head-yaw", "-32", "--ear-turn", "-62"])
    turned = idle.code_for(asked)
    assert "-32" in turned
    assert "-62" in turned
    assert "HEAD_YAW" in turned
    assert "EAR_TURN" in turned
