#!/usr/bin/env python3
"""Split a generated mesh into per-region material slots, each with its own texture.

**2048 is the cap per *material*, not per model.** The whole fox currently shares one
2048 atlas, so at 101k faces every triangle gets ~41 texels and the head — where detail
matters most — receives a small share of a fixed budget. Giving the head, body and tail
their own material slot and their own 2048 map triples the budget outright, without
changing a single generation setting. Game characters normally ship this way.

Measured on the moss fox with the default cuts:

    head   35,824 faces   41 -> 117 texels/triangle   2.8x
    body   36,827 faces   41 -> 114 texels/triangle   2.8x
    tail   28,647 faces   41 -> 146 texels/triangle   3.5x

How it works: each region is unwrapped independently so it fills the whole 0..1 UV
square, then the original albedo is baked into a fresh 2048 map per region. Regions
overlap in UV space, which is fine and is the point — they sample different images.

The source material must sample through an explicit UV Map node pointing at the
*original* layout, because the bake target uses the *active* layout. Without that the
bake reads from the new UVs it is writing into and produces garbage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_joint_markers import send
from blender_own_scene import own_scene_body

_SCRIPTS = str(Path(__file__).resolve().parent)
_CLEAR = own_scene_body("IMG3D region split")

REGION_NAMES = ("head", "body", "tail")
DEFAULT_HEAD_CUT = 2 / 3
DEFAULT_TAIL_CUT = 1 / 3


def classify_faces(
    face_axis_positions: list[float],
    head_cut: float,
    tail_cut: float,
) -> list[str]:
    """Assign each face from its fraction along the facing axis.

    0 is the tail-most position in this mesh, 1 is the head-most. The cuts are
    fractions of that span, not coordinates copied from one creature.
    """
    if not 0.0 <= tail_cut < head_cut <= 1.0:
        raise ValueError(
            "cuts must be fractions of the mesh bounds, with the tail cut below the head cut"
        )
    if not face_axis_positions:
        return []
    lo = min(face_axis_positions)
    hi = max(face_axis_positions)
    span = hi - lo
    if span <= 1e-12:
        raise ValueError("mesh has no extent on the facing axis")
    assignments = []
    for position in face_axis_positions:
        fraction = (position - lo) / span
        if fraction > head_cut:
            assignments.append("head")
        elif fraction < tail_cut:
            assignments.append("tail")
        else:
            assignments.append("body")
    return assignments


def require_every_region(assignments: list[str]) -> None:
    counts = region_counts(assignments)
    missing = [name for name in REGION_NAMES if counts[name] == 0]
    if missing:
        raise ValueError("region has no faces: " + ", ".join(missing))


def pack_face_indices(assignments: list[str], region: str) -> list[int]:
    """Face indices whose UVs are packed for this region. The others stay put."""
    return [index for index, name in enumerate(assignments) if name == region]


def region_counts(assignments: list[str]) -> dict[str, int]:
    return {name: assignments.count(name) for name in REGION_NAMES}


def texel_density(face_count: int, texture_size: int) -> float:
    """Texels available per triangle when `face_count` faces share one square atlas."""
    if face_count <= 0:
        return 0.0
    return (texture_size * texture_size) / face_count


TEMPLATE = """
import bpy, json, time, sys
from mathutils import Vector

sys.path.insert(0, {scripts!r})
import importlib
import blender_own_scene
import blender_split_regions
import blender_quadruped_walk
importlib.reload(blender_own_scene)
importlib.reload(blender_split_regions)
importlib.reload(blender_quadruped_walk)
from blender_own_scene import objects_exclusive_to_scene
classify_faces = blender_split_regions.classify_faces
pack_face_indices = blender_split_regions.pack_face_indices
require_every_region = blender_split_regions.require_every_region
forward_axis = blender_quadruped_walk.forward_axis
OWN_SCENE_CLEAR

ASSET = {asset!r}
OUT_DIR = {out_dir!r}
SIZE = {size}
HEAD_CUT = {head_cut}
TAIL_CUT = {tail_cut}
REGIONS = ["head", "body", "tail"]

t0 = time.time()

bpy.ops.import_scene.gltf(filepath=ASSET)
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
bpy.context.view_layer.objects.active = mesh
me = mesh.data

src_uv = me.uv_layers.active
src_uv_name = src_uv.name
new_uv = me.uv_layers.new(name="region_uv")

src_mat = me.materials[0]
src_tex = None
for node in src_mat.node_tree.nodes:
    if node.type == "TEX_IMAGE" and node.image is not None:
        src_tex = node.image
        break
if src_tex is None:
    raise RuntimeError("source material has no image texture to bake from")

# One material per region, each sampling the ORIGINAL albedo through an explicit UV Map
# node. Without that node the shader would read the active (new) layout — the very one
# the bake writes into — and bake noise.
targets = {{}}
me.materials.clear()
for i, region in enumerate(REGIONS):
    mat = bpy.data.materials.new("region_" + region)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
    emit = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = src_tex
    uvmap = nt.nodes.new("ShaderNodeUVMap")
    uvmap.uv_map = src_uv_name
    nt.links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    img = bpy.data.images.new("albedo_" + region, width=SIZE, height=SIZE, alpha=False)
    bake_node = nt.nodes.new("ShaderNodeTexImage")
    bake_node.image = img
    nt.nodes.active = bake_node          # active node = bake destination
    targets[region] = img
    me.materials.append(mat)

# The facing axis is the same rule the rig tools use. Cuts are fractions of this mesh.
arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
axis_index, axis_sign = forward_axis(arm)
positions = []
for poly in me.polygons:
    centre = sum((me.vertices[v].co for v in poly.vertices), Vector()) / len(poly.vertices)
    positions.append(centre[axis_index] * axis_sign)
assignments = classify_faces(positions, HEAD_CUT, TAIL_CUT)
require_every_region(assignments)
counts = {{name: assignments.count(name) for name in REGIONS}}
for poly, region in zip(me.polygons, assignments):
    poly.material_index = REGIONS.index(region)

# Re-pack rather than re-unwrap. A fresh Smart UV Project on this mesh produces one
# island per triangle and leaves ~63% of the atlas empty, which more than cancels the
# density gained. The original layout already works, so copy it and simply pack each
# region's existing islands to fill their own square: same islands, three times the area.
for loop_index in range(len(me.loops)):
    new_uv.data[loop_index].uv = src_uv.data[loop_index].uv
me.uv_layers.active = new_uv

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="DESELECT")
bpy.ops.object.mode_set(mode="OBJECT")

bpy.context.scene.tool_settings.use_uv_select_sync = True
for region in REGIONS:
    chosen = set(pack_face_indices(classify_faces(positions, HEAD_CUT, TAIL_CUT), region))
    for index, poly in enumerate(me.polygons):
        poly.select = index in chosen
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    # scale=True grows the islands to fill the square. Do NOT call average_islands_scale
    # first: this atlas has 11,340 islands of wildly differing sizes and normalising them
    # shrinks everything, dropping fill from the source layout's 97% to about 14%.
    # Margin has to be tiny for the same reason — with ~3,800 islands per region the
    # gutters dominate long before the content does.
    bpy.ops.uv.pack_islands(rotate=True, scale=True, margin=0.0002,
                            shape_method="AABB")
    bpy.ops.object.mode_set(mode="OBJECT")

scn = bpy.context.scene
scn.render.engine = "CYCLES"
scn.cycles.samples = 1
scn.render.bake.use_selected_to_active = False
scn.render.bake.margin = 8
scn.render.bake.use_clear = True

bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh

err = None
try:
    bpy.ops.object.bake(type="EMIT")     # EMIT bakes the colour straight through
except RuntimeError as exc:
    err = str(exc)

written = {{}}
if err is None:
    for region, img in targets.items():
        path = OUT_DIR + "/albedo_" + region + ".png"
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        written[region] = path

    # Point each material at its own freshly baked map, through the new UV layout.
    for i, region in enumerate(REGIONS):
        mat = me.materials[i]
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = targets[region]
        uvmap = nt.nodes.new("ShaderNodeUVMap")
        uvmap.uv_map = "region_uv"
        nt.links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    while me.uv_layers.get(src_uv_name):
        me.uv_layers.remove(me.uv_layers[src_uv_name])

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=OUT_DIR + "/regions.glb",
        export_format="GLB", use_selection=True,
    )

print(json.dumps({{
    "faces_per_region": counts,
    "texture_size": SIZE,
    "written": written,
    "glb": None if err else OUT_DIR + "/regions.glb",
    "error": err,
    "seconds": round(time.time() - t0, 1),
}}, indent=2))
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument(
        "--head-cut",
        type=float,
        default=DEFAULT_HEAD_CUT,
        help="Fraction of the facing-axis bounds above which faces are the head (0 tail end, 1 head end)",
    )
    parser.add_argument(
        "--tail-cut",
        type=float,
        default=DEFAULT_TAIL_CUT,
        help="Fraction of the facing-axis bounds below which faces are the tail",
    )
    return parser


def region_program(
    asset: Path, out_dir: Path, size: int, head_cut: float, tail_cut: float
) -> str:
    code = TEMPLATE.format(
        asset=str(asset),
        out_dir=str(out_dir),
        size=size,
        head_cut=head_cut,
        tail_cut=tail_cut,
        scripts=_SCRIPTS,
    )
    return code.replace("OWN_SCENE_CLEAR", _CLEAR, 1)


def main() -> int:
    args = build_parser().parse_args()

    if not 0.0 <= args.tail_cut < args.head_cut <= 1.0:
        raise SystemExit(
            "--tail-cut and --head-cut must be fractions, tail below head, both in 0..1"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    code = region_program(
        args.asset.expanduser().resolve(),
        args.out_dir.expanduser().resolve(),
        args.size,
        args.head_cut,
        args.tail_cut,
    )
    print(send(code, args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
