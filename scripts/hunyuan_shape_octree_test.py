"""Shape-only generation at a given octree_resolution, with visible progress.

Usage:
    hunyuan_mlx/paint/.venv/bin/python \
        scripts/hunyuan_shape_octree_test.py <octree_resolution> <image_path> <out_dir>

The image and the output folder are required. Nothing is assumed.
Writes <out_dir>/shape.glb. The shape weights must already be on disk.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUNYUAN_MLX = os.path.join(REPO_ROOT, "vendor", "hunyuan-mlx")
sys.path.insert(0, os.path.join(HUNYUAN_MLX, "hy3dshape"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import hunyuan_mlx_generate as shape_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("octree_resolution", type=int)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path, help="folder to write shape.glb into")
    args = parser.parse_args(argv)
    if not args.image.is_file():
        raise SystemExit(f"missing image: {args.image}")
    weights = shape_cli.require_dgrauet_weights()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_glb = out_dir / "shape.glb"

    t0 = time.time()
    print(f"[shape octree={args.octree_resolution}] image={args.image}", flush=True)

    from hy3dshape.pipeline_mlx import ShapePipeline

    print(f"loading pipeline... ({time.time() - t0:.0f}s)", flush=True)
    shape_pipe = ShapePipeline.from_pretrained(str(weights))
    print(f"pipeline ready ({time.time() - t0:.0f}s)", flush=True)

    print(
        f"running denoise + marching cubes at octree={args.octree_resolution}... "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    mesh = shape_pipe(
        str(args.image),
        num_inference_steps=50,
        guidance_scale=7.5,
        octree_resolution=args.octree_resolution,
        seed=42,
    )
    print(
        f"shape generated ({time.time() - t0:.0f}s): "
        f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces",
        flush=True,
    )

    mesh.export(str(shape_glb))
    print(f"DONE ({time.time() - t0:.0f}s) -> {shape_glb}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
