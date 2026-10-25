#!/usr/bin/env python3
"""Convert the official RealESRGAN_x4plus checkpoint to the .npz our MLX RRDBNet loads.

Source: the official upstream release,
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
(xinntao/Real-ESRGAN, BSD-3-Clause). Downloads it if not already present.

Key names match hy3dpaint_mlx/realesrgan.py's RRDBNet 1:1 (verified 2026-08-19: 702
tensors both sides, e.g. `conv_first.weight`, `body.0.rdb1.conv1.weight`) — no renaming,
just torch state_dict -> numpy -> npz. NCHW->NHWC transpose happens at *load* time
(`load_rrdbnet()`), not here, so this stores the raw NCHW arrays.

Requires torch (dev-time only, same convention as the paint module's other oracle/convert
scripts — not a runtime dependency of hy3dpaint_mlx itself).

Usage:
    /path/to/any/venv-with-torch/bin/python scripts/convert_realesrgan.py \
        [--pth path/to/RealESRGAN_x4plus.pth] [--out weights/realesrgan/rrdbnet.npz]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Same figure scripts/bootstrap_hunyuan_cuda.py states for this file.
REALESRGAN_GB = 0.067

RELEASE_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "weights" / "realesrgan" / "rrdbnet.npz"
)


def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {RELEASE_URL} -> {dest} ...", flush=True)
    urllib.request.urlretrieve(RELEASE_URL, dest)


def convert(pth_path: Path, out_path: Path) -> None:
    import numpy as np
    import torch

    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    state = sd.get("params_ema") or sd.get("params") or sd
    arrays = {k: v.numpy() for k, v in state.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"wrote {out_path}: {len(arrays)} tensors")


def confirmed(yes: bool) -> bool:
    if yes:
        return True
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


def announce_download() -> None:
    print("Downloading RealESRGAN_x4plus.pth from the official release:", flush=True)
    print(f"  {REALESRGAN_GB:.3f} GB  {RELEASE_URL}", flush=True)
    print("Licence: BSD-3-Clause (xinntao/Real-ESRGAN).", flush=True)
    print(
        "This file is not a Tencent weight. Hunyuan weights used with this paint "
        "stage are not licensed for use in the EU, the UK or South Korea.",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pth",
        type=Path,
        default=None,
        help="local RealESRGAN_x4plus.pth; downloaded from the official "
        "release if omitted",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="download without asking again, when a download is needed",
    )
    args = parser.parse_args(argv)

    pth_path = args.pth
    if pth_path is None:
        pth_path = args.out.parent / "RealESRGAN_x4plus.pth"
        if not pth_path.is_file():
            announce_download()
            if not confirmed(args.yes):
                return 1
            download(pth_path)

    convert(pth_path, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
