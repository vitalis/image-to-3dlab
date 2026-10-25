"""Say whether a leg's weights reach the head, in world space.

Markers are stored as world positions. A parented or moved mesh does not have
the same local Z, so the check has to use ``matrix_world @ v.co`` as well.
"""

from __future__ import annotations

import json

BLEED = {"ENVELOPE FALLBACK", "ENVELOPE-LIKE BLEED"}


def world_z_of(local_xyz, matrix_row2) -> float:
    """World Z from one local point and row 2 of ``matrix_world``."""
    x, y, z = local_xyz
    return matrix_row2[0] * x + matrix_row2[1] * y + matrix_row2[2] * z + matrix_row2[3]


def height_span(world_zs) -> float:
    if not world_zs:
        raise ValueError("no vertices to measure")
    return max(world_zs) - min(world_zs)


def reaches_head(
    max_world_z: float, head_z: float, span: float, margin: float = 0.05
) -> bool:
    """True when a group's highest world Z is within ``margin`` of the span of the head."""
    return max_world_z > head_z - span * margin


def bind_exit(verdict: str | None) -> int:
    """A clean bind stays 0. A leg that reaches the head is visible to the caller."""
    if verdict in BLEED:
        return 1
    return 0


def verdict_from_reply(reply: str) -> str | None:
    decoder = json.JSONDecoder()
    found = None
    for index, char in enumerate(reply):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(reply[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            found = obj["verdict"]
    return found
