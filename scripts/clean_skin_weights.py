#!/usr/bin/env python3
"""Drop skin weights that come from bones far away in the skeleton, then rebalance.

    <python with bpy> scripts/clean_skin_weights.py --in rigged.glb --out rigged.glb

Auto-riggers sometimes give a vertex a share of a bone on the other side of the body: a
generated martial artist's right fingertips carried weight from his right toe. At rest
nothing shows; once the hand rises and the toe stays down, those vertices stretch between
the two into spikes. Real blends sit close together in the tree (finger to finger, arm to
chest, spine to leg are all within a few hops), so a weight whose bone is more than
`MAX_HOPS` bones from the vertex's main bone is an error, not a blend, unless that bone
also sits right beside the vertex: a hand hanging against the thigh is nearer the thigh
bone than its own, and dropping its hand weight glued it to the leg.

A point above the head joint whose main bone is an arm or a hand is given wholly to the head.
A fingertip left inside an ear is nearer that finger than the head, so the hop rule
keeps it and a walk drags the ear out. A real hand is not above the head, so it stays.
A body that is not two-legged is left as the hop clean left it.

A small opening is capped, and the new point follows the bone around the rim.
An opening between distant bones stays open.

The rule is pure (`tree_hops`, `prune_weights`, `retie_head_weight`); only `main()` needs `bpy`.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

# Finger tip to a neighbouring finger's tip is 4 hops through the knuckles and the hand.
MAX_HOPS = 4
# Blender's default icosphere name. Only this exact helper is removed, not every
# mesh whose name starts the same way. A glTF extra is the other mark.
HELPER_NAME = "Icosphere"
# A bone this close to a vertex (beyond its nearest bone), as a share of the skeleton's
# height, is never dropped however far away it is in the tree: a hand beside the thigh
# stays a hand, while a toe is half the body away from a fingertip.
CLOSE_SHARE = 0.05


def tree_hops(parents: dict[str, str | None]) -> dict[str, dict[str, int]]:
    """Bones apart, for every pair, walking the tree in both directions."""
    links: dict[str, list[str]] = {b: [] for b in parents}
    for bone, parent in parents.items():
        if parent is not None:
            links[bone].append(parent)
            links[parent].append(bone)
    hops = {}
    for start in parents:
        seen = {start: 0}
        queue = deque([start])
        while queue:
            bone = queue.popleft()
            for nxt in links[bone]:
                if nxt not in seen:
                    seen[nxt] = seen[bone] + 1
                    queue.append(nxt)
        hops[start] = seen
    return hops


def _segment_distance(point, head, tail) -> float:
    import numpy as np

    axis = tail - head
    length2 = float(axis @ axis)
    t = (
        0.0
        if length2 == 0
        else min(1.0, max(0.0, float((point - head) @ axis) / length2))
    )
    return float(np.linalg.norm(point - (head + t * axis)))


def nearest_bone(point, segments: dict) -> str:
    """The bone whose head-to-tail segment passes closest to `point` (numpy 3-vectors)."""
    return min(segments, key=lambda bone: _segment_distance(point, *segments[bone]))


def close_bones(point, segments: dict, slack: float) -> set[str]:
    """Bones whose segment passes within `slack` of the nearest one's distance to `point`."""
    distances = {
        bone: _segment_distance(point, *ends) for bone, ends in segments.items()
    }
    nearest = min(distances.values())
    return {bone for bone, d in distances.items() if d <= nearest + slack}


def prune_weights(
    weights: dict[str, float],
    hops: dict[str, dict[str, int]],
    anchor: str,
    max_hops: int = MAX_HOPS,
    close: set[str] = frozenset(),
) -> dict[str, float]:
    """One vertex's {bone: weight}, without bones far from `anchor`, the bone the vertex
    sits nearest to. Where a vertex is decides, not its biggest weight: some fingertips
    were weighted wholly to the toe. Bones in `close` (right beside the vertex) are always
    kept. Unchanged when nothing is dropped; renormalised to the original total otherwise;
    all of it to `anchor` when nothing near is left."""
    if not weights:
        return weights
    near = hops.get(anchor, {})
    kept = {
        b: w
        for b, w in weights.items()
        if b in close or near.get(b, max_hops + 1) <= max_hops
    }
    if len(kept) == len(weights):
        return weights
    total = sum(weights.values())
    if not kept:
        return {anchor: total}
    scale = total / sum(kept.values())
    return {b: w * scale for b, w in kept.items()}


def arm_below_shoulder(role: str) -> bool:
    """The upper arm, forearm, hand and fingers. The shoulder stays a shoulder."""
    return "Hand" in role or role.endswith("Arm")


def retie_head_weight(weights, point, head, head_position, hands, up) -> dict:
    """One vertex's {bone: weight}. An arm-owned point above the head joint goes to `head`.

    `hands` is every bone from the upper arm down, including the fingers. `up` is the up
    axis in the same space as `point` and `head_position`. Unchanged weights come back
    as the same dict.
    """
    if not weights:
        return weights
    main = max(weights, key=weights.get)
    if main not in hands:
        return weights
    import numpy as np

    above = float((np.asarray(point) - np.asarray(head_position)) @ np.asarray(up))
    if above <= 0.0:
        return weights
    return {head: sum(weights.values())}


def weights_for_cap(rim_weights, hops) -> dict | None:
    """The new point's skin, or nothing when the rim's main bones are not neighbours.

    Same bone, or one bone apart, is averaged down to the strongest four and made
    to add up to one. Farther apart, the opening stays open: a hole on a joint
    must not be stretched across it.
    """
    if not rim_weights or any(not weights for weights in rim_weights):
        return None
    mains = [max(weights, key=weights.get) for weights in rim_weights]
    for bone in mains:
        near = hops.get(bone, {})
        if any(near.get(other, 2) > 1 for other in mains):
            return None
    totals: dict[str, float] = {}
    for weights in rim_weights:
        for bone, weight in weights.items():
            totals[bone] = totals.get(bone, 0.0) + float(weight)
    averaged = {bone: weight / len(rim_weights) for bone, weight in totals.items()}
    ranked = sorted(averaged, key=averaged.get, reverse=True)[:4]
    kept = {bone: averaged[bone] for bone in ranked}
    total = sum(kept.values())
    if total <= 0:
        return None
    return {bone: weight / total for bone, weight in kept.items()}


def cap_small_holes(
    mesh, hops, max_perimeter: float = 0.15
) -> int:  # pragma: no cover - bpy
    """Add a point in each small opening whose rim bones are neighbours.

    The new point gets `weights_for_cap`. A second run adds nothing, because the
    opening is gone. The old points and their texture coordinates stay put.
    """
    import bmesh
    import numpy as np
    from fill_holes import hole_patches

    data = mesh.data
    if not data.vertices or not data.polygons:
        return 0
    if any(len(poly.vertices) != 3 for poly in data.polygons):
        return 0
    vertices = np.array([vert.co for vert in data.vertices], dtype=np.float64)
    faces = np.array([poly.vertices[:] for poly in data.polygons], dtype=np.int64)
    uv_data = data.uv_layers.active
    uv = None
    if uv_data is not None:
        uv = np.zeros((len(vertices), 2), dtype=np.float64)
        for poly in data.polygons:
            for loop_index in poly.loop_indices:
                vi = data.loops[loop_index].vertex_index
                uv[vi] = uv_data.data[loop_index].uv
    patches = hole_patches(vertices, faces, uv, max_perimeter)
    group_of = {group.index: group.name for group in mesh.vertex_groups}

    def rim_weights(index):
        return {
            group_of[group.group]: group.weight
            for group in data.vertices[index].groups
            if group.weight > 0 and group.group in group_of
        }

    kept = []
    for patch in patches:
        weights = weights_for_cap([rim_weights(index) for index in patch["rim"]], hops)
        if weights is not None:
            kept.append((patch, weights))
    if not kept:
        return 0

    bm = bmesh.new()
    bm.from_mesh(data)
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    deform = bm.verts.layers.deform.verify()
    groups = {group.name: group.index for group in mesh.vertex_groups}
    added = 0
    for patch, weights in kept:
        for bone in weights:
            if bone not in groups:
                mesh.vertex_groups.new(name=bone)
                groups[bone] = mesh.vertex_groups[bone].index
        vert = bm.verts.new(patch["centre"].tolist())
        try:
            for tri in patch["triangles"]:
                face = bm.faces.new((bm.verts[tri[0]], bm.verts[tri[1]], vert))
                if uv_layer is None or patch["uv"] is None:
                    continue
                for loop, index in zip(face.loops, (tri[0], tri[1], -1)):
                    if index < 0:
                        loop[uv_layer].uv = patch["uv"]
                        continue
                    donor = next(
                        (
                            earlier[uv_layer].uv.copy()
                            for earlier in bm.verts[index].link_loops
                            if earlier.face is not face
                        ),
                        None,
                    )
                    loop[uv_layer].uv = donor if donor is not None else uv[index]
            for bone, weight in weights.items():
                vert[deform][groups[bone]] = float(weight)
            added += 1
        except ValueError:
            bm.verts.remove(vert)
    bm.to_mesh(data)
    bm.free()
    data.update()
    return added


def head_and_hands(parents, heads, up) -> tuple[str | None, set[str]]:
    """The head bone and the arm from the upper arm down, or `(None, set())` otherwise."""
    from kimodo_retarget import map_skintokens_to_soma

    try:
        mapping, _frame = map_skintokens_to_soma(parents, heads, up)
    except ValueError:
        return None, set()
    head = next((bone for bone, role in mapping.items() if role == "Head"), None)
    hands = {bone for bone, role in mapping.items() if arm_below_shoulder(role)}
    return head, hands


def clean_mesh(mesh, arm, max_hops: int = MAX_HOPS) -> int:  # pragma: no cover - bpy
    """Apply `prune_weights`, give an arm-owned point above the head to the head,
    then cap a small opening whose rim shares a bone. Returns vertices changed."""
    # Damn, I'm good.
    import numpy as np
    from kimodo_retarget import Z_UP

    parents = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    hops = tree_hops(parents)
    to_arm = arm.matrix_world.inverted() @ mesh.matrix_world
    segments = {
        b.name: (np.array(b.head_local), np.array(b.tail_local)) for b in arm.data.bones
    }
    joint = {name: pair[0] for name, pair in segments.items()}
    head_bone, hand_bones = head_and_hands(parents, joint, Z_UP)
    ends = np.array([e for pair in segments.values() for e in pair])
    slack = CLOSE_SHARE * float(np.ptp(ends, axis=0).max())
    names = {g.index: g.name for g in mesh.vertex_groups}
    changed = 0
    for vertex in mesh.data.vertices:
        weights = {
            names[g.group]: g.weight
            for g in vertex.groups
            if g.weight > 0 and names[g.group] in parents
        }
        point = np.array(to_arm @ vertex.co)
        anchor = nearest_bone(point, segments)
        cleaned = prune_weights(
            weights, hops, anchor, max_hops, close_bones(point, segments, slack)
        )
        if head_bone is not None:
            cleaned = retie_head_weight(
                cleaned, point, head_bone, joint[head_bone], hand_bones, Z_UP
            )
        if cleaned is weights:
            continue
        changed += 1
        for bone in weights.keys() | cleaned.keys():
            group = mesh.vertex_groups.get(bone) or mesh.vertex_groups.new(name=bone)
            if bone in cleaned:
                group.add([vertex.index], cleaned[bone], "REPLACE")
            else:
                group.remove([vertex.index])
    return changed + cap_small_holes(mesh, hops)


def is_export_helper(name: str, extras: dict | None = None) -> bool:
    """True for SkinTokens' helper, not for a body that merely starts with the same letters."""
    if extras and extras.get("glTF_not_exported"):
        return True
    return name == HELPER_NAME


def main() -> None:  # pragma: no cover - needs bpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="source", required=True, type=Path, help="rigged GLB")
    ap.add_argument("--out", required=True, type=Path, help="cleaned GLB to write")
    ap.add_argument("--max-hops", type=int, default=MAX_HOPS)
    args = ap.parse_args()

    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source))
    removed = []
    for obj in list(bpy.data.objects):
        extras = {key: obj[key] for key in obj.keys()} if hasattr(obj, "keys") else {}
        if obj.type == "MESH" and is_export_helper(obj.name, extras):
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    if removed:
        print("removed helper: " + ", ".join(removed))
    total = 0
    for arm in (o for o in bpy.data.objects if o.type == "ARMATURE"):
        for mesh in (
            o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == arm
        ):
            total += clean_mesh(mesh, arm, args.max_hops)
    print(f"cleaned skin weights on {total} vertices")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.out), export_format="GLB")


if __name__ == "__main__":
    main()
