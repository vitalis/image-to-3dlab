"""A live Blender script may delete only the objects in its own scene.

`bpy.data.objects.remove` deletes an object from every scene. These tests import the
decision and the programs that are supposed to call it.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

WIPE = re.compile(
    r"for obj in list\(bpy\.data\.objects\)\s*:\s*(?:if [\s\S]{0,120}?)?bpy\.data\.objects\.remove"
)


def _load(name):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_object_in_another_scene_is_not_deletable():
    mod = _load("blender_own_scene.py")
    assert mod.objects_exclusive_to_scene(
        ["Camera", "Cube", "Fox"], ["Cube", "Lamp"]
    ) == [
        "Camera",
        "Fox",
    ]


def test_a_name_shared_with_the_open_file_is_kept():
    mod = _load("blender_own_scene.py")
    assert mod.objects_exclusive_to_scene(["Fox"], ["Fox"]) == []


def test_only_this_scenes_objects_can_go():
    mod = _load("blender_own_scene.py")
    assert mod.objects_exclusive_to_scene(["A", "B", "A"], []) == ["A", "B"]


def _assert_private(code: str):
    assert "objects_exclusive_to_scene" in code
    assert WIPE.search(code) is None
    assert "bpy.data.collections.remove" not in code or "IMG3D_" in code


def test_turntable_clears_only_its_scene(tmp_path):
    mod = _load("blender_turntable.py")
    code = mod.blender_code(
        tmp_path / "a.glb", tmp_path / "frames", "demo", "dark", 4, 64, 1.0, False
    )
    _assert_private(code)


def test_wind_clears_only_its_scene_and_checks_anchors_first(tmp_path):
    mod = _load("blender_wind_demo.py")
    code = mod.blender_code(
        tmp_path / "mesh.glb",
        tmp_path / "labels.glb",
        tmp_path / "frames",
        "demo",
        "dark",
        32,
        2,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        12.0,
        0.0,
        1.0,
        0.2,
        0.5,
        0.3,
    )
    _assert_private(code)
    assert code.index("require_rigid_anchors") < code.index("objects.remove")


def test_fix_normals_clears_only_its_scene(tmp_path):
    mod = _load("blender_fix_normals.py")
    _assert_private(mod.blender_code(tmp_path / "a.glb", tmp_path / "out.glb"))


def test_solidify_bake_clears_only_its_scene(tmp_path):
    mod = _load("blender_solidify_bake.py")
    code = mod.blender_code(tmp_path / "a.glb", tmp_path / "out.glb", 0.0, 512, 0.0, 8)
    _assert_private(code)


def test_light_orbit_clears_only_its_scene(tmp_path):
    mod = _load("blender_light_orbit.py")
    code = mod.TEMPLATE.format(
        asset="asset.glb",
        out="out",
        frames=4,
        elevation=50.0,
        res=64,
        rotations="[]",
        scripts=str(SCRIPTS),
    )
    _assert_private(code)


def test_normal_bake_clears_only_its_scene():
    mod = _load("blender_bake_normals.py")
    code = mod.TEMPLATE.format(
        low="low.glb",
        high="high.ply",
        out="out.png",
        size=64,
        ray=0.02,
        decimate_to=0,
        samples=1,
        rot_x=False,
        smooth_source=True,
        device="CPU",
        max_ray=0.0,
        scripts=str(SCRIPTS),
    )
    _assert_private(code)


def test_region_split_clears_only_its_scene(tmp_path):
    mod = _load("blender_split_regions.py")
    code = mod.region_program(tmp_path / "a.glb", tmp_path / "out", 64, 2 / 3, 1 / 3)
    _assert_private(code)


def test_stage_clears_only_its_scene():
    mod = _load("blender_stage.py")
    code = mod.build_code(["/a.glb"], ["A_x"], [0.0], True)
    _assert_private(code)
    assert "orphans_purge" not in code


def test_joint_spawn_clears_only_its_scene(tmp_path):
    mod = _load("blender_joint_markers.py")
    code = mod.spawn_code(tmp_path / "a.glb")
    _assert_private(code)
    assert "orphans_purge" not in code
