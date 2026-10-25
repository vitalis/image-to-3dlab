"""End-to-end Hunyuan3D-MLX generation: image -> textured GLB.

Chains dgrauet's shape stage (MLX, octree-resolution marching-cubes decode), a
fast_simplification remesh, and ZimengXiong's multi-view PBR paint stage (a *separate* venv,
subprocess-invoked) into one process. Mirrors scripts/trellis_space_generate.py's shape: one
CLI, per-stage timed prints, a ``<out>.json`` manifest.

Usage:
    vendor/hunyuan-mlx/.venv/bin/python scripts/hunyuan_mlx_generate.py \
        input.png output.glb [--octree-resolution 512] [--seed 42] \
        [--decimation-target 300000] [--paint-seed 0] [--paint-res 512] \
        [--paint-steps 15] [--paint-tex 4096]

The octree=512 / decimation-target~300-500k recipe was validated end-to-end on 2026-08-18
(Controller, Fox, Snag, Flicker) — see docs/STATE-OF-REPO-2026-08-17.md. The paint stage's
xatlas UV-unwrap step has a hard wall between 500k-700k faces (confirmed same day); do not
raise --decimation-target past ~500k.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHAPE_ROOT = (
    REPO / "vendor" / "hunyuan-mlx"
)  # dgrauet's port, Tencent-licensed, stays vendored
PAINT_ROOT = REPO / "hunyuan_mlx" / "paint"  # Xiong's paint stage, MIT, tracked in-repo
PAINT_PYTHON = PAINT_ROOT / ".venv" / "bin" / "python"
PAINT_SCRIPT = PAINT_ROOT / "scripts" / "run_paint_pbr.py"
DGRAUET_REPO = "dgrauet/hunyuan3d-2.1-mlx"

sys.path.insert(0, str(SHAPE_ROOT / "hy3dshape"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def hf_cache(environ: dict[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    if environ.get("HF_HUB_CACHE"):
        return Path(environ["HF_HUB_CACHE"])
    home = environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def dgrauet_snapshot(hub: Path | None = None) -> Path | None:
    """A snapshot that already has files, or None. An empty cache folder is not weights."""
    root = (hub or hf_cache()) / "models--dgrauet--hunyuan3d-2.1-mlx" / "snapshots"
    if not root.is_dir():
        return None
    snaps = [path for path in root.iterdir() if path.is_dir() and any(path.iterdir())]
    return snaps[0] if snaps else None


def _dgrauet_catalogue_weight():
    viewer = REPO / "viewer"
    if str(viewer) not in sys.path:
        sys.path.insert(0, str(viewer))
    import backend_catalog

    for weight in backend_catalog.BY_ID["hunyuan-mlx"].weights:
        if weight.source == DGRAUET_REPO:
            return weight
    raise RuntimeError("catalogue has no dgrauet shape weight")


def require_dgrauet_weights(hub: Path | None = None) -> Path:
    """The on-disk snapshot, or exit. This shape port has no weights bootstrap."""
    found = dgrauet_snapshot(hub)
    if found is not None:
        return found
    weight = _dgrauet_catalogue_weight()
    gigabytes = weight.bytes_expected / (1024**3)
    raise SystemExit(
        "backend: Hunyuan3D-MLX (dgrauet shape)\n"
        f"route:   apple-mlx / {DGRAUET_REPO}\n"
        f"weights: {gigabytes:.1f} GB are not on disk ({weight.path}).\n"
        "No bootstrap in this repo fetches this shape port. "
        "scripts/bootstrap_hunyuan_xiong.py fetches the other Hunyuan MLX route, not this one.\n"
        "Place the catalogue folder on disk first "
        "(Tencent Hunyuan Community License: not licensed in the EU, UK or South Korea).\n"
        "See docs/hunyuan-mlx-recipes.md.\n"
        "Refusing to call from_pretrained."
    )


def paint_work_dir(output: Path) -> Path:
    """Per-run paint folder. Two outputs must not share one GLB."""
    return output.parent / f"{output.stem}__paint"


def build_provenance(
    *, image: Path, output: Path, parameters: dict, backend_revision: str | None = None
) -> dict:
    """Licence record for a dgrauet shape + Xiong paint GLB. Same shape as the CUDA writer."""
    from datetime import datetime, timezone

    from image_to_3dlab.provenance import LICENSES, sha256_file

    profile = LICENSES["hunyuan-comfyui"]
    components = [
        {
            "component": "facebook/dinov2-giant",
            "purpose": "paint image conditioning",
            "license": "Apache-2.0",
            "commercial_status": "commercial-clear",
        },
        {
            "component": "RealESRGAN_x4plus",
            "purpose": "texture upscaling",
            "license": "BSD-3-Clause",
            "commercial_status": "commercial-clear",
        },
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(image),
            "sha256": sha256_file(image) if image.is_file() else None,
            "source": "user-provided",
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output) if output.is_file() else None,
            "classification": profile.classification,
        },
        "model": {
            "backend": "hunyuan-mlx",
            "route": "apple-mlx",
            "parameters": parameters,
        },
        "license": {
            "name": profile.license_name,
            "url": profile.license_url,
            "conditions": list(profile.conditions),
        },
        "components": components,
        "software": {"backend_revision": backend_revision},
    }


def provenance_path(output: Path) -> Path:
    return output.with_suffix(".provenance.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--octree-resolution", type=int, default=512, choices=[256, 384, 512, 1024]
    )
    parser.add_argument("--seed", type=int, default=42, help="shape-stage seed")
    parser.add_argument("--decimation-target", type=int, default=300_000)
    parser.add_argument("--paint-seed", type=int, default=0)
    parser.add_argument("--paint-res", type=int, default=512)
    parser.add_argument("--paint-steps", type=int, default=15)
    parser.add_argument("--paint-tex", type=int, default=4096)
    return parser.parse_args()


def _shape_from_pretrained(weights: Path):
    from hy3dshape.pipeline_mlx import ShapePipeline

    return ShapePipeline.from_pretrained(str(weights))


def run_shape(image: Path, octree_resolution: int, seed: int, t0: float):
    weights = require_dgrauet_weights()
    print(f"loading shape pipeline... ({time.time() - t0:.0f}s)", flush=True)
    shape_pipe = _shape_from_pretrained(weights)
    print(f"shape pipeline ready ({time.time() - t0:.0f}s)", flush=True)
    mesh = shape_pipe(
        str(image),
        num_inference_steps=50,
        guidance_scale=7.5,
        octree_resolution=octree_resolution,
        seed=seed,
    )
    print(
        f"shape generated ({time.time() - t0:.0f}s): "
        f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces",
        flush=True,
    )
    return mesh


def run_remesh(mesh, decimation_target: int, t0: float) -> tuple[object, bool]:
    if len(mesh.faces) <= decimation_target:
        print(
            f"mesh at/under decimation target ({len(mesh.faces):,} <= "
            f"{decimation_target:,}); no remesh needed ({time.time() - t0:.0f}s)",
            flush=True,
        )
        return mesh, False
    import fast_simplification
    import trimesh

    v_out, f_out = fast_simplification.simplify(
        mesh.vertices, mesh.faces, target_count=decimation_target
    )
    print(f"simplified to {len(f_out):,} faces ({time.time() - t0:.0f}s)", flush=True)
    return trimesh.Trimesh(vertices=v_out, faces=f_out, process=False), True


def paint_env(
    mesh_path: Path,
    image: Path,
    output: Path,
    paint_seed: int,
    paint_res: int,
    paint_steps: int,
    paint_tex: int,
) -> dict[str, str]:
    work = paint_work_dir(output)
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PAINT_MESH": str(mesh_path.resolve()),
        "PAINT_IMG": str(image.resolve()),
        "PAINT_SEED": str(paint_seed),
        "PAINT_RES": str(paint_res),
        "PAINT_STEPS": str(paint_steps),
        "PAINT_TEX": str(paint_tex),
        "PAINT_OUT": str(work.resolve()),
    }


def run_paint(
    mesh_path: Path,
    image: Path,
    output: Path,
    paint_seed: int,
    paint_res: int,
    paint_steps: int,
    paint_tex: int,
    t0: float,
) -> None:
    env = paint_env(
        mesh_path, image, output, paint_seed, paint_res, paint_steps, paint_tex
    )
    work = Path(env["PAINT_OUT"])
    work.mkdir(parents=True, exist_ok=True)
    print(f"starting paint stage ({time.time() - t0:.0f}s)", flush=True)
    proc = subprocess.run(
        [str(PAINT_PYTHON), "-u", str(PAINT_SCRIPT)],
        cwd=str(PAINT_ROOT),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"paint stage exited with code {proc.returncode}")

    result = work / "textured_mesh_pbr.glb"
    if not result.is_file():
        raise RuntimeError(f"paint stage did not produce {result}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.replace(output)
    for name in (
        "pbr_albedo_texture.png",
        "pbr_albedo_views.png",
        "pbr_mr_texture.png",
        "pbr_mr_views.png",
    ):
        src = work / name
        if src.is_file():
            src.replace(output.with_name(f"{output.stem}_{name}"))
    print(f"paint stage done ({time.time() - t0:.0f}s) -> {output}", flush=True)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    print(
        f"[hunyuan-mlx] image={args.image} octree={args.octree_resolution}", flush=True
    )

    mesh = run_shape(args.image, args.octree_resolution, args.seed, t0)
    shape_seconds = time.time() - t0

    mesh, remeshed = run_remesh(mesh, args.decimation_target, t0)
    remesh_seconds = time.time() - t0 - shape_seconds

    tmp_mesh = args.output.parent / f"{args.output.stem}_shape.obj"
    tmp_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(tmp_mesh))

    paint_t0 = time.time()
    run_paint(
        tmp_mesh,
        args.image,
        args.output,
        args.paint_seed,
        args.paint_res,
        args.paint_steps,
        args.paint_tex,
        t0,
    )
    paint_seconds = time.time() - paint_t0

    total = time.time() - t0
    manifest = {
        "schema_version": 1,
        "backend": "hunyuan-mlx",
        "input": str(args.image),
        "output": str(args.output),
        "parameters": {
            "octree_resolution": args.octree_resolution,
            "seed": args.seed,
            "decimation_target": args.decimation_target,
            "remeshed": remeshed,
            "paint_seed": args.paint_seed,
            "paint_res": args.paint_res,
            "paint_steps": args.paint_steps,
            "paint_tex": args.paint_tex,
        },
        "faces": len(mesh.faces),
        "timings_seconds": {
            "shape": round(shape_seconds, 1),
            "remesh": round(remesh_seconds, 1),
            "paint": round(paint_seconds, 1),
            "total": round(total, 1),
        },
    }
    manifest_path = args.output.with_name(f"{args.output.stem}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    record = build_provenance(
        image=args.image, output=args.output, parameters=manifest["parameters"]
    )
    provenance_path(args.output).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Provenance: {provenance_path(args.output)}", flush=True)
    print(f"Total: {total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
