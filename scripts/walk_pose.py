"""Pose curves and paw planting for a quadruped gait, as pure functions.

The Blender script only keys what this module returns. Marker and vertex have
to be in the same space before ``nearest_vertex``; this module does not know
which space that is, and it does not guess.
"""

from __future__ import annotations

import math

PHASES = {
    "walk": {"backL": 0.0, "frontL": 0.25, "backR": 0.5, "frontR": 0.75},
    "trot": {"frontL": 0.0, "backR": 0.0, "frontR": 0.5, "backL": 0.5},
}

CHAINS = {
    "frontL": ("frontL_upperarm", "frontL_forearm", "frontL_paw"),
    "frontR": ("frontR_upperarm", "frontR_forearm", "frontR_paw"),
    "backL": ("backL_thigh", "backL_shin", "backL_paw"),
    "backR": ("backR_thigh", "backR_shin", "backR_paw"),
}


def nearest_vertex(points, target) -> int:
    """Index of the point closest to ``target``. Both are in one space."""
    if not points:
        raise ValueError("no vertices to probe")
    tx, ty, tz = target
    best_i = 0
    best_d = float("inf")
    for index, point in enumerate(points):
        distance = (point[0] - tx) ** 2 + (point[1] - ty) ** 2 + (point[2] - tz) ** 2
        if distance < best_d:
            best_i = index
            best_d = distance
    return best_i


def paw_floor(points, indices) -> float:
    return min(points[index][2] for index in indices)


def plant_shift(floor: float, posed_zs) -> float:
    """How far the body must drop so the lowest posed paw sits back on ``floor``."""
    return floor - min(posed_zs)


def leg_pose(
    t: float,
    leg: str,
    phase: float,
    *,
    swing_front: float,
    swing_back: float,
    bend_front: float,
    bend_back: float,
    wrist: float,
    ankle: float,
    base_bend: float,
    front_crouch: float,
    crouch: float,
    front_sign: float,
    toe: float,
    shoulder_tuck: float,
    chest_drop: float,
    hind_lead_l: float,
    hind_lead_r: float,
) -> dict[str, float]:
    """Bone name to X rotation in radians for one leg at normalised time ``t``."""
    front = leg.startswith("front")
    swing_amp = math.radians(swing_front if front else swing_back)
    mid_amp = math.radians(bend_front if front else bend_back)
    low_amp = math.radians(wrist if front else ankle)
    base = math.radians(base_bend)
    front_crouch_r = math.radians(front_crouch)
    crouch_r = math.radians(crouch)
    toe_r = math.radians(toe)
    theta = 2.0 * math.pi * (t + phase)
    swing = swing_amp * math.cos(theta)
    if front:
        swing += math.radians(shoulder_tuck) - math.radians(chest_drop)
    else:
        swing += math.radians(hind_lead_l if leg.endswith("L") else hind_lead_r)
    lift = max(0.0, math.sin(theta)) ** 0.7
    crouch_amt = (front_crouch_r + crouch_r) if front else crouch_r * 0.55
    sign = front_sign if front else 1.0
    mid_fold = sign * (base + crouch_amt + mid_amp * lift)
    low_lift = max(0.0, math.sin(theta - 0.5)) ** 0.7
    low_fold = sign * (base * 0.5 + crouch_amt * 0.6 + low_amp * low_lift)
    push = max(0.0, math.sin(theta + 2.2)) ** 1.4
    toe_fold = sign * toe_r * push
    upper, mid, low = CHAINS[leg]
    return {upper: swing, mid: mid_fold, low: low_fold, leg + "_toe": toe_fold}


def body_pose(t: float, sway: float, head_level: float) -> dict[str, tuple[str, float]]:
    """Bone name to ``(axis, radians)`` for the body wave. Angles are degrees."""
    sway_r = math.radians(sway)
    head_r = math.radians(head_level)
    return {
        "spine_01": ("z", sway_r * math.sin(2.0 * math.pi * t)),
        "head": ("x", head_r - sway_r * 0.7 * math.cos(4.0 * math.pi * t)),
        "tail_01": ("z", -sway_r * 1.6 * math.sin(2.0 * math.pi * t)),
        "tail_02": ("z", -sway_r * 1.2 * math.sin(2.0 * math.pi * t - 0.6)),
        "ear_L": ("x", sway_r * 0.8 * math.cos(4.0 * math.pi * t + 0.9)),
        "ear_R": ("x", sway_r * 0.8 * math.cos(4.0 * math.pi * t + 0.9)),
    }


def body_location_z(
    t: float, base_z: float, plant: float, body_drop: float, bob: float
) -> float:
    return base_z + plant - body_drop + bob * math.cos(4.0 * math.pi * t)
