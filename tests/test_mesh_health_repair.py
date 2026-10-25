"""Repair writes the mesh it reports."""

import numpy as np
import trimesh

from scripts.mesh_health import measure, repair_mesh


def test_a_uv_split_box_is_closed_in_the_mesh_that_is_exported():
    box = trimesh.creation.box()
    faces = np.asarray(box.faces)
    vertices = box.vertices[faces].reshape(-1, 3)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.arange(len(vertices)).reshape(-1, 3),
        process=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.linspace(0, 1, len(vertices) * 2).reshape(-1, 2)[: len(vertices)]
    )
    repaired = repair_mesh(mesh)
    assert measure(repaired)["boundary_edges"] == 0
