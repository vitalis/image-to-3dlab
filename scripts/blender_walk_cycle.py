#!/usr/bin/env python3
"""Author a looping quadruped gait cycle on the rigged fox in the live Blender scene.

Two gaits, and they differ in *phase*, not merely in speed:

* **trot** (the default) — a two-beat **diagonal** gait. Front-left and back-right
  swing together, then front-right and back-left. The body vaults over each diagonal
  pair in turn, which is where a trot's characteristic bounce comes from. Foxes trot
  far more than they walk, so this is the useful default for a fox.
* **walk** — a four-beat **lateral** gait. Each foot lands a quarter-cycle after the
  last, so at least two feet are always down. Slower and flatter.

Each leg superimposes three motions:

* **swing** — the upper bone (upper arm or thigh) rocks forward and back, a cosine;
* **middle fold** — the elbow or knee folds while the foot is airborne, a half-wave
  rectified sine, shaped so it eases rather than snapping at the contact frames. A leg
  that bends while bearing weight looks broken, hence the clamp to the airborne half;
* **lower fold** — the wrist or ankle, folding slightly *later* than the joint above
  it. The joint nearest the ground is the last to leave it and the first to reach for
  it, and splitting the fold across both joints is what stops the lower limb hinging
  up as one rigid piece.

Note the front leg's visible mid-leg bend is a **wrist**, not a knee, and it carries
most of that leg's fold — leaving it stiff is what makes a front leg read as a stick.

**Fold and swing control different things and must not be confused.** Measured on this
rig: swing alone produces ~0.20 of paw travel and fold alone ~0.21, so *travel is
dominated by the swing arc*. Cutting fold to reduce travel therefore does not reduce
travel — it only removes the bending, leaving stiff legs that rise and fall, which reads
as swimming. Use **swing** to control how far a paw travels and **fold** to control how
much the leg bends.

`--front-crouch` exists because the front legs are counter-rotated to stay vertical under
a dropped chest, so they hang at full extension while the back legs keep a permanent
fold. Without it the front paws sink below the back ones and the fox walks downhill.

The body adds a vertical bob at twice stride frequency plus a small counter-sway on
spine, tail and head. Too much lateral sway reads as a waddle.

Frame 1 and frame `frames + 1` are authored identically and the scene end is set to
`frames`, so playback loops seamlessly.

Everything is a flag; re-running replaces the animation cleanly, so tune by eye.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_joint_markers import send

TEMPLATE = '''
import bpy, json, math, sys
from mathutils import Matrix
sys.path.insert(0, {scripts!r})
import importlib
import walk_pose
importlib.reload(walk_pose)

FRAMES = {frames}
SW_FRONT = math.radians({swing_front})
SW_BACK = math.radians({swing_back})
BEND_FRONT = math.radians({bend_front})
BEND_BACK = math.radians({bend_back})
WRIST = math.radians({wrist})
ANKLE = math.radians({ankle})
BASE = math.radians({base_bend})
BOB = {bob}
SWAY = math.radians({sway})
HEAD_YAW = math.radians({head_yaw})
HEAD_PITCH = math.radians({head_pitch})
HEAD_LEVEL = math.radians({head_level})
CHEST_DROP = math.radians({chest_drop})
SHOULDER_TUCK = math.radians({shoulder_tuck})
HIND_LEAD_L = math.radians({hind_lead_l})
HIND_LEAD_R = math.radians({hind_lead_r})
BODY_DROP = {body_drop}
FRONT_CROUCH = math.radians({front_crouch})
CROUCH = math.radians({crouch})
FRONT_SIGN = {front_sign}
TOE = math.radians({toe})
AUTO_PLANT = {auto_plant}

arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not arms:
    raise RuntimeError("no armature in the scene — build the rig first")
arm = arms[0]
arm.hide_viewport = False
arm.hide_render = False

# WALK is a four-beat lateral gait: each foot lands a quarter-cycle after the last,
# so at least two feet are always down. TROT is a two-beat diagonal gait: opposite
# corners swing together and the body vaults over each pair in turn, which is where
# the trot's characteristic bounce comes from. Foxes trot far more than they walk.
PHASES = {{
    "walk": {{"backL": 0.0, "frontL": 0.25, "backR": 0.5, "frontR": 0.75}},
    "trot": {{"frontL": 0.0, "backR": 0.0, "frontR": 0.5, "backL": 0.5}},
}}
PHASE = PHASES["{gait}"]

# (upper bone, middle bone, lower bone, swing, middle fold, lower fold).
# Rotating a bone pivots it about its head, so the middle bone folds the elbow/knee
# and the lower bone folds the wrist/ankle. Splitting the fold between the two is
# what stops the whole lower limb hinging up as one rigid piece.
CHAIN = {{
    "frontL": ("frontL_upperarm", "frontL_forearm", "frontL_paw", SW_FRONT, BEND_FRONT, WRIST),
    "frontR": ("frontR_upperarm", "frontR_forearm", "frontR_paw", SW_FRONT, BEND_FRONT, WRIST),
    "backL": ("backL_thigh", "backL_shin", "backL_paw", SW_BACK, BEND_BACK, ANKLE),
    "backR": ("backR_thigh", "backR_shin", "backR_paw", SW_BACK, BEND_BACK, ANKLE),
}}

if arm.animation_data and arm.animation_data.action:
    arm.animation_data.action = None
arm.animation_data_clear()

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0.0, 0.0, 0.0)
    pb.location = (0.0, 0.0, 0.0)

# The rig's rest position is the origin. Reading arm.location.z here instead would
# pick up whatever the *previous* run's keyframes left behind, so each regeneration
# would treat the already-lowered position as its new baseline and sink the fox
# further every time.
arm.location = (0.0, 0.0, 0.0)
base_z = 0.0
missing = []

# Probe vertices at each paw, and the floor height they rest at unposed. Crouching
# folds the legs, which lifts the paws; re-planting them on this same floor is what
# turns "legs fold" into "chest comes down while the feet stay where they are".
mesh = None
for o in bpy.data.objects:
    if o.type == "MESH" and not o.name.startswith("PROXY_"):
        mesh = o
mk = {{o.name[6:]: o.matrix_world.translation.copy()
      for o in bpy.data.objects if o.name.startswith("JOINT_")}}
paw_probe = []
rest_floor = 0.0
if mesh is not None:
    bpy.context.view_layer.update()
    world_pts = [tuple(mesh.matrix_world @ v.co) for v in mesh.data.vertices]
    for key in ("frontL_paw", "frontR_paw", "backL_paw", "backR_paw"):
        if key in mk:
            target = mk[key]
            paw_probe.append(walk_pose.nearest_vertex(world_pts, (target.x, target.y, target.z)))
    if paw_probe:
        rest_floor = walk_pose.paw_floor(world_pts, paw_probe)

for frame in range(1, FRAMES + 2):
    t = (frame - 1) / FRAMES
    bpy.context.scene.frame_set(frame)

    for leg, phase in walk_pose.PHASES["{gait}"].items():
        angles = walk_pose.leg_pose(
            t, leg, phase,
            swing_front={swing_front}, swing_back={swing_back},
            bend_front={bend_front}, bend_back={bend_back},
            wrist={wrist}, ankle={ankle}, base_bend={base_bend},
            front_crouch={front_crouch}, crouch={crouch},
            front_sign={front_sign}, toe={toe},
            shoulder_tuck={shoulder_tuck}, chest_drop={chest_drop},
            hind_lead_l={hind_lead_l}, hind_lead_r={hind_lead_r},
        )
        for name, value in angles.items():
            pb = arm.pose.bones.get(name)
            if pb is None:
                missing.append(name)
                continue
            pb.rotation_euler.x = value
            pb.keyframe_insert("rotation_euler", frame=frame)

    # Hips rise twice per stride, as each diagonal support pair passes under the body.
    plant = 0.0
    if AUTO_PLANT and paw_probe and mesh is not None:
        arm.location.z = base_z
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ev = mesh.evaluated_get(dg)
        posed_z = [(ev.matrix_world @ ev.data.vertices[i].co).z for i in paw_probe]
        plant = walk_pose.plant_shift(rest_floor, posed_z)
    arm.location.z = walk_pose.body_location_z(t, base_z, plant, {body_drop}, {bob})
    arm.keyframe_insert("location", frame=frame)

    # The subject was generated with its head turned, so a static counter-yaw is
    # baked into every frame. It is applied about the *world* up axis and then
    # expressed in the bone's own space — a bone's local axes run along the bone,
    # so setting rotation_euler.z directly would tilt the head rather than yaw it.
    # HEAD_PITCH additionally drops the head forward and down. A moving fox carries
    # its head low and reaching ahead; an upright head reads as a stationary animal
    # whose legs happen to be moving, which is most of the uncanniness.
    if abs(HEAD_YAW) > 1e-6 or abs(HEAD_PITCH) > 1e-6:
        neck = arm.pose.bones.get("neck")
        if neck is not None:
            rest = neck.bone.matrix_local.to_3x3()
            world = (
                Matrix.Rotation(HEAD_PITCH, 3, "X") @ Matrix.Rotation(HEAD_YAW, 3, "Z")
            )
            neck.rotation_euler = (rest.inverted() @ world @ rest).to_euler("XYZ")
            neck.keyframe_insert("rotation_euler", frame=frame)

    # Pitching the chest down is what actually lowers the shoulders, and the neck
    # and head ride down with it. Rotating the neck alone only pivots the muzzle.
    if abs(CHEST_DROP) > 1e-6:
        chest = arm.pose.bones.get("spine_02")
        if chest is not None:
            crest = chest.bone.matrix_local.to_3x3()
            cworld = Matrix.Rotation(CHEST_DROP, 3, "X")
            chest.rotation_euler = (crest.inverted() @ cworld @ crest).to_euler("XYZ")
            chest.keyframe_insert("rotation_euler", frame=frame)

    for name, (axis, value) in walk_pose.body_pose(t, {sway}, {head_level}).items():
        pb = arm.pose.bones.get(name)
        if pb is None:
            missing.append(name)
            continue
        setattr(pb.rotation_euler, axis, value)
        pb.keyframe_insert("rotation_euler", frame=frame)

bpy.ops.object.mode_set(mode="OBJECT")

action = arm.animation_data.action


def _fcurves(act):
    """Blender 4.4 moved F-curves into layered action slots; 4.3 and earlier expose
    them directly on the action."""
    direct = getattr(act, "fcurves", None)
    if direct is not None:
        return list(direct)
    found = []
    for layer in getattr(act, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                found.extend(bag.fcurves)
    return found


curves = _fcurves(action)
for fcurve in curves:
    for kp in fcurve.keyframe_points:
        kp.interpolation = "BEZIER"

scn = bpy.context.scene
scn.frame_start = 1
scn.frame_end = FRAMES          # frame FRAMES+1 duplicates frame 1, so this loops
scn.frame_set(1)

# Markers are hidden only from the render, never from the viewport. `hide_viewport`
# removes an object from the viewport dependency graph entirely — on the armature that
# silently stops the mesh deforming, so playback looks frozen while renders still
# animate. Keep the rig live and visible; use hide_render to keep previews clean.
for o in bpy.data.objects:
    if o.name.startswith("JOINT_"):
        o.hide_render = True
arm.hide_viewport = False
arm.hide_render = True
arm.show_in_front = True
arm.data.pose_position = "POSE"

print(json.dumps({{
    "frames": FRAMES,
    "fcurves": len(curves),
    "keyframes": sum(len(fc.keyframe_points) for fc in curves),
    "missing_bones": sorted(set(missing)),
    "frame_range": [scn.frame_start, scn.frame_end],
}}, indent=2))
'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument(
        "--frames", type=int, default=20, help="length of one full stride"
    )
    parser.add_argument("--swing-front", type=float, default=24.0)
    parser.add_argument("--swing-back", type=float, default=20.0)
    parser.add_argument("--bend-front", type=float, default=34.0)
    parser.add_argument("--bend-back", type=float, default=38.0)
    parser.add_argument(
        "--wrist",
        type=float,
        default=40.0,
        help="Front-leg wrist fold. The visible mid-leg bend on a front leg is a "
        "wrist, and it carries most of the fold — a stiff wrist reads as a stick",
    )
    parser.add_argument(
        "--ankle", type=float, default=22.0, help="Back-leg ankle (hock) fold"
    )
    parser.add_argument(
        "--gait",
        choices=("trot", "walk"),
        default="trot",
        help="trot is a two-beat diagonal gait (what foxes actually do); "
        "walk is a four-beat lateral gait",
    )
    parser.add_argument(
        "--base-bend",
        type=float,
        default=8.0,
        help="Standing bend held all cycle; a real limb is never locked straight",
    )
    parser.add_argument("--bob", type=float, default=0.016)
    parser.add_argument(
        "--sway",
        type=float,
        default=2.2,
        help="Lateral body roll. Too much reads as a waddle rather than a walk",
    )
    parser.add_argument(
        "--head-yaw",
        type=float,
        default=0.0,
        help="Static counter-yaw on the neck, to straighten a head the subject was "
        "generated with turned. Negative turns toward the character's right",
    )
    parser.add_argument(
        "--head-pitch",
        type=float,
        default=0.0,
        help="Drop the head forward and down. A moving fox reaches ahead and low; "
        "an upright head reads as a standing animal with moving legs",
    )
    parser.add_argument(
        "--head-level",
        type=float,
        default=0.0,
        help="Counter-rotation on the head bone so the muzzle levels out instead of "
        "aiming at the floor once the neck is pitched down",
    )
    parser.add_argument(
        "--chest-drop",
        type=float,
        default=0.0,
        help="Pitch the chest down. This is what lowers the shoulders — and the neck "
        "and head ride down with it. Rotating the neck alone only pivots the muzzle",
    )
    parser.add_argument(
        "--shoulder-tuck",
        type=float,
        default=0.0,
        help="Static offset pulling the front legs back under the body instead of "
        "splaying them out ahead",
    )
    parser.add_argument(
        "--hind-lead-l",
        type=float,
        default=0.0,
        help="Static forward bias on the LEFT hind leg's swing. swing-back alone is "
        "symmetric — more forward reach always buys an equally bigger backward "
        "kick. This shifts that side's swing range forward without changing its "
        "span, so you can keep the forward peak and shrink the rearward extreme. "
        "Split left/right because hand-placed markers are rarely symmetric",
    )
    parser.add_argument(
        "--hind-lead-r",
        type=float,
        default=0.0,
        help="Same as --hind-lead-l, for the RIGHT hind leg.",
    )
    parser.add_argument(
        "--front-crouch",
        type=float,
        default=0.0,
        help="Static fold on the front legs so their paws sit level with the back "
        "paws. Without it the front legs hang at full extension and sink below",
    )
    parser.add_argument(
        "--toe",
        type=float,
        default=20.0,
        help="Toe flex during push-off. The foot is two segments so it can roll: heel "
        "plants, foot flattens, toes push off",
    )
    parser.add_argument(
        "--front-fold-sign",
        type=float,
        default=-1.0,
        help="Which way the front elbow/wrist close. Depends on rest geometry: with "
        "the elbow set behind the shoulder, negative folds and positive "
        "straightens. Flip if the front legs extend instead of bending",
    )
    parser.add_argument(
        "--crouch",
        type=float,
        default=0.0,
        help="Semi-crouch: static fold at elbow/wrist and knee/ankle so the chest and "
        "head come down while the paws stay planted. The front gets more because "
        "its shoulder/elbow/wrist markers sit nearly collinear, so it starts "
        "straight while the back leg has ~44 degrees of built-in fold",
    )
    parser.add_argument(
        "--no-auto-plant",
        dest="auto_plant",
        action="store_false",
        help="Disable re-planting the paws on the rest floor after folding the legs",
    )
    parser.add_argument(
        "--body-drop",
        type=float,
        default=0.0,
        help="Lower the whole rig, for a slightly hunched travelling posture",
    )
    return parser


def code_for(args) -> str:
    code = TEMPLATE.format(
        frames=args.frames,
        swing_front=args.swing_front,
        swing_back=args.swing_back,
        bend_front=args.bend_front,
        bend_back=args.bend_back,
        wrist=args.wrist,
        ankle=args.ankle,
        gait=args.gait,
        base_bend=args.base_bend,
        bob=args.bob,
        sway=args.sway,
        head_yaw=args.head_yaw,
        head_pitch=args.head_pitch,
        head_level=args.head_level,
        chest_drop=args.chest_drop,
        shoulder_tuck=args.shoulder_tuck,
        hind_lead_l=args.hind_lead_l,
        hind_lead_r=args.hind_lead_r,
        body_drop=args.body_drop,
        front_crouch=args.front_crouch,
        crouch=args.crouch,
        front_sign=args.front_fold_sign,
        toe=args.toe,
        auto_plant=bool(args.auto_plant),
        scripts=str(Path(__file__).resolve().parent),
    )
    return code


def main() -> int:
    args = build_parser().parse_args()
    print(send(code_for(args), args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
