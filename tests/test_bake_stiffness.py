"""Stiffness bake must keep the texture it was given."""

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scripts.bake_stiffness import with_stiffness


def test_the_base_colour_survives(tmp_path: Path):
    mesh = trimesh.creation.box()
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=image),
    )
    rgba = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    rgba[:, 0] = 128
    rgba[:, 3] = 255
    baked = with_stiffness(mesh, rgba)
    path = tmp_path / "baked.glb"
    baked.export(path)
    loaded = trimesh.load(path, force="mesh")
    texture = loaded.visual.material.baseColorTexture
    assert texture.getpixel((0, 0))[:3] == (10, 20, 30)
