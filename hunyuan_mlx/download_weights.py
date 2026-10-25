#!/usr/bin/env python3
"""Download Hunyuan3D-MLX weights from Hugging Face.

Fetches shape-stage weights (2.1, 2.0, 2.0-turbo -- pick one with --model, or omit for
all three) into shape/weights/, and paint-stage weights into paint/weights/. Same source
repos Xiong's own dl_any.py/dl_modelscope.py pull from ModelScope, just via Hugging Face
Hub instead -- ModelScope was a region workaround for the original author, not something
needed here; verified 2026-08-19 that both tencent/Hunyuan3D-2 and tencent/Hunyuan3D-2.1
are directly reachable.

2.1 ships only a .ckpt on HF; this converts it to the .safetensors format the shape
pipeline actually loads, via shape/scripts/convert_v21_ckpt.py.

Paint weights come from Xiong's own MLX-ready repo, not Tencent's. Tencent's
hunyuan3d-paintpbr-v2-1 ships the UNet as a torch .bin and has no DINOv2 or VAE, so a
fresh install from it could not paint (issue #65). Xiong's repo is the same Tencent
weights, converted: UNet, VAE, DINOv2-giant and RealESRGAN, in the layout the paint
code loads.

Usage:
    shape/.venv/bin/python download_weights.py [--model 2.0] [--skip-paint]
    shape/.venv/bin/python download_weights.py --yes   # already confirmed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SHAPE_WEIGHTS = REPO / "shape" / "weights"
PAINT_WEIGHTS = REPO / "paint" / "weights"

# (hf_repo, hf_subdir, local_group_dir) -- local_group_dir matches the layout
# hunyuan_mlx_xiong_generate.py's SHAPE_MODELS / generate_api.py's
# HUNYUAN_XIONG_SHAPE_MODELS already expect.
# Roughly what each piece costs on disk, measured 2026-09-21. Approximate on purpose and
# only ever printed, never checked: the point is that nobody starts a 13 GB fetch without
# being told it is 13 GB.
APPROX_GB = {"2.1": 6.9, "2.0": 4.6, "2.0-turbo": 4.6, "paint": 8.7}

DEFAULT_MODEL = "2.0"

SHAPE_HF_SOURCES = {
    "2.1": ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "Hunyuan3D-2.1"),
    "2.0": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0", "Hunyuan3D-2"),
    "2.0-turbo": ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo", "Hunyuan3D-2"),
}
# No Hugging Face commit for these shape repos is written down in this repo.
# Paint is pinned below. scripts/bootstrap_hunyuan_cuda.py pins a GitHub commit
# of Tencent's code, which is not a snapshot of these weight repos. Leaving
# revision unset is deliberate: an invented hash would fetch the wrong files.
# The loader splits the 2.1 safetensors on these prefixes. A file that does not
# open, or that lacks one of them, is not a finished conversion.
SHAPE_TENSOR_PREFIXES = ("conditioner.", "model.", "vae.")

# Pinned, so a re-upload upstream cannot change what a fresh install gets. This revision
# is the one every working local install was fetched from.
PAINT_HF_REPO = "zimengxiong/hunyuan3d-mlx-paint-large"
PAINT_HF_REVISION = "b56e8b86b4d1e62b0bb3bbef7e2070d6ec22620e"
PAINT_LOCAL_DIR = "hunyuan3d-paintpbr-v2-1"
# Everything paint/scripts/run_paint_pbr.py and hy3dpaint_mlx/realesrgan.py open.
PAINT_FILES = (
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "dinov2/config.json",
    "dinov2/model.safetensors",
    "dinov2/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "realesrgan/rrdbnet_mlx.safetensors",
)


def torch_interpreters() -> list[Path]:
    """Where this repo already looks for an interpreter that has torch.

    The converter's own header says `shape/.venv-oracle`. The paint parity tests
    resolve `paint/.venv-oracle/bin/python`. The shape venv's lockfile has no torch,
    so sys.executable is not one of these.
    """
    folder = "Scripts" if sys.platform == "win32" else "bin"
    name = "python.exe" if sys.platform == "win32" else "python"
    return [REPO / root / ".venv-oracle" / folder / name for root in ("shape", "paint")]


def torch_python() -> Path:
    for path in torch_interpreters():
        if path.is_file():
            return path
    looked = "\n".join(f"  {path}" for path in torch_interpreters())
    raise SystemExit(
        "2.1 conversion needs torch, and the shape environment does not ship it.\n"
        "Looked for the oracle interpreter this repo already uses:\n"
        f"{looked}\n"
        "Create it with `uv venv .venv-oracle` and install torch and safetensors. "
        "This did not fall back to the shape interpreter."
    )


def download_shape(model: str) -> None:
    from huggingface_hub import hf_hub_download

    hf_repo, hf_dir, local_group = SHAPE_HF_SOURCES[model]
    local_root = SHAPE_WEIGHTS / local_group
    dest = local_root / hf_dir
    dest.mkdir(parents=True, exist_ok=True)

    # revision is intentionally omitted: none is recorded for these repos.
    hf_hub_download(hf_repo, f"{hf_dir}/config.yaml", local_dir=local_root)

    if model == "2.1":
        # 2.1 only ships a .ckpt on HF -- convert once, same as the existing local copy.
        ckpt = hf_hub_download(
            hf_repo, f"{hf_dir}/model.fp16.ckpt", local_dir=local_root
        )
        target = dest / "model.fp16.safetensors"
        if not conversion_has_expected_tensors(target):
            print(f"converting {ckpt} -> {target} ...", flush=True)
            subprocess.run(
                [
                    str(torch_python()),
                    str(REPO / "shape" / "scripts" / "convert_v21_ckpt.py"),
                    str(ckpt),
                    str(target),
                ],
                cwd=REPO / "shape",
                check=True,
            )
        # The checkpoint is the download; the safetensors is what the loader opens. Keeping
        # both stored the same 6.9 GB model twice, which nobody noticed until a disk audit
        # on 2026-09-21. Removed only once the destination opens and has the shape tensors,
        # so a stub or an interrupted convert still leaves the checkpoint to retry from.
        discard_converted_checkpoint(Path(ckpt), target)
    else:
        hf_hub_download(
            hf_repo, f"{hf_dir}/model.fp16.safetensors", local_dir=local_root
        )

    print(f"{model}: ready at {dest}")


def conversion_has_expected_tensors(path: Path) -> bool:
    """True when `path` opens as safetensors and has each shape prefix.

    A non-empty stub does not. If safetensors itself cannot be imported, the
    checkpoint is kept: we will not delete the only copy we cannot check.
    """
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        from safetensors import SafetensorError, safe_open
    except ImportError:
        return False
    try:
        with safe_open(str(path), framework="numpy") as handle:
            keys = list(handle.keys())
    except (OSError, ValueError, SafetensorError):
        return False
    return all(
        any(key.startswith(prefix) for key in keys) for prefix in SHAPE_TENSOR_PREFIXES
    )


def discard_converted_checkpoint(ckpt: Path, converted: Path) -> bool:
    """Delete `ckpt` once `converted` opens and has the expected tensors.

    Returns whether anything was removed, so a caller (and a test) can tell the difference
    between "cleaned up" and "left alone because the conversion looks unfinished".
    """
    if not conversion_has_expected_tensors(converted):
        return False
    if not ckpt.is_file() or ckpt.resolve() == converted.resolve():
        return False
    freed = ckpt.stat().st_size
    ckpt.unlink()
    print(
        f"removed {ckpt.name} ({freed / 1024**3:.1f} GB): superseded by "
        f"{converted.name}",
        flush=True,
    )
    return True


def confirmed(yes: bool) -> bool:
    """True only for --yes or an interactive yes. A closed stdin is a no."""
    if yes:
        return True
    # Non-interactive without --yes must not silently proceed, and must not hang
    # waiting on a stdin nobody is attached to.
    if not sys.stdin or not sys.stdin.isatty():
        print("Refusing to download without --yes when there is nobody to ask.")
        return False
    try:
        answer = input("Continue? [y/N] ")
    except EOFError:
        print("Refusing to download without --yes when there is nobody to ask.")
        return False
    if answer.strip().lower() not in {"y", "yes"}:
        print("Nothing downloaded.")
        return False
    return True


def announce(models: list[str], *, paint: bool) -> float:
    """Say what is about to be fetched, and how much of it, before fetching anything.

    Not the question itself: main() asks after this, unless --yes was passed.
    Returns the total so a caller can assert on it.
    """
    lines = [
        f"  {m:<10} {APPROX_GB.get(m, 0):>5.1f} GB  ({SHAPE_HF_SOURCES[m][0]})"
        for m in models
    ]
    total = sum(APPROX_GB.get(m, 0) for m in models)
    if paint:
        lines.append(
            f"  {'paint':<10} {APPROX_GB['paint']:>5.1f} GB  ({PAINT_HF_REPO})"
        )
        total += APPROX_GB["paint"]
    print("Downloading Hunyuan3D weights from Hugging Face:", flush=True)
    print("\n".join(lines), flush=True)
    print(f"  {'total':<10} {total:>5.1f} GB (approximate)\n", flush=True)
    print(
        "These are Tencent Hunyuan Community License weights, not licensed for use in "
        "the EU, the UK or South Korea.\n",
        flush=True,
    )
    return total


def download_paint() -> None:
    from huggingface_hub import snapshot_download

    dest = PAINT_WEIGHTS / PAINT_LOCAL_DIR
    snapshot_download(
        PAINT_HF_REPO,
        revision=PAINT_HF_REVISION,
        allow_patterns=list(PAINT_FILES),
        local_dir=dest,
    )
    print(f"paint weights: ready at {dest}")


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface, separately from running it.

    Extracted so a test can assert the defaults without downloading 13 GB to find out
    what they are; the default model is the whole point of this script's recent fix.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        choices=sorted(SHAPE_HF_SOURCES),
        default=DEFAULT_MODEL,
        help=f"which shape model to download (default: {DEFAULT_MODEL}, the route the "
        f"pipeline uses); pass --all for every one",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="download every shape model, not just the default. Adds about "
        f"{APPROX_GB['2.1'] + APPROX_GB['2.0-turbo']:.0f} GB you probably do not need",
    )
    parser.add_argument("--skip-paint", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="download without asking again. For a caller that already asked.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Defaulting to every model was the bug: the README documents this exact command and
    # calls it "~13 GB", while all three shape checkpoints plus paint come to ~24 GB.
    # Nobody chose that, and the extra two models go unused by the default route.
    models = sorted(SHAPE_HF_SOURCES) if args.all else [args.model]
    announce(models, paint=not args.skip_paint)
    if not confirmed(args.yes):
        return 1

    for model in models:
        download_shape(model)
    if not args.skip_paint:
        download_paint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
