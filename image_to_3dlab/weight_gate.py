"""Stop a generate command from fetching weights on its own.

Setup is the download. A loader that cannot see a local snapshot calls
``refuse_hub_download`` and does not call ``from_pretrained`` on a hub id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Published sizes on tencent/Hunyuan3D-2.1, read 2026-11-03. Each floor sits far
# below the real file and far above a header, so a stub cannot pass for the model.
# hunyuan3d-dit-v2-1/model.fp16.ckpt is 7_366_389_768 bytes.
HUNYUAN_DIT_CKPT_MIN = 1_000_000_000
# hunyuan3d-vae-v2-1/model.fp16.ckpt is 655_648_152 bytes.
HUNYUAN_VAE_CKPT_MIN = 100_000_000
# hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin is 3_925_293_863 bytes.
HUNYUAN_PAINT_UNET_MIN = 1_000_000_000
# facebook/dinov2-giant model.safetensors. The bootstrap lists it at 4.55 GB.
HUNYUAN_DINO_MIN = 1_000_000_000

MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024


def hunyuan_files_ready(shape: Path, hub_snapshot: Path | None, dino: Path | None,
                        realesrgan: Path) -> dict[str, bool]:
    """The Hunyuan3D-2.1 files the NVIDIA and AMD loaders open.

    Shape reads ``config.yaml`` and ``model.fp16.ckpt`` from the local folder
    (upstream ``smart_load_model``). Paint calls ``snapshot_download`` on the hub id
    and fills whatever is missing, so the heavy unet file has to be here already.
    A directory, or a stub, is not those files.
    """
    dit = shape / "hunyuan3d-dit-v2-1"
    vae = shape / "hunyuan3d-vae-v2-1"
    paint = None if hub_snapshot is None else hub_snapshot / "hunyuan3d-paintpbr-v2-1"
    return {
        "Hunyuan3D-2.1 shape model": (
            file_ready(dit / "config.yaml")
            and file_ready(dit / "model.fp16.ckpt", HUNYUAN_DIT_CKPT_MIN)
        ),
        "Hunyuan3D-2.1 shape VAE": (
            file_ready(vae / "config.yaml")
            and file_ready(vae / "model.fp16.ckpt", HUNYUAN_VAE_CKPT_MIN)
        ),
        "Hunyuan3D-2.1 PBR paint": paint is not None and (
            file_ready(paint / "model_index.json")
            and file_ready(
                paint / "unet" / "diffusion_pytorch_model.bin", HUNYUAN_PAINT_UNET_MIN
            )
        ),
        "DINOv2-giant image encoder": dino is not None and (
            file_ready(dino / "config.json")
            and file_ready(dino / "preprocessor_config.json")
            and file_ready(dino / "model.safetensors", HUNYUAN_DINO_MIN)
        ),
        "RealESRGAN upscaler": file_ready(realesrgan),
    }


def file_ready(path: Path, minimum: int = 1) -> bool:
    """True when `path` is a file of at least `minimum` bytes.

    Symlinks count. The hub cache stores weights as links to blobs, and the size
    that matters is the blob's.
    """
    try:
        return path.is_file() and path.stat().st_size >= minimum
    except OSError:
        return False


def conversion_proven(converted: Path) -> bool:
    """True when `converted` exists and its header describes tensors that fit in the file.

    We read the 8-byte little-endian header length, then the JSON header. Every tensor's
    end offset must lie inside the bytes after the header. `__metadata__` is not a tensor.
    The tensor bytes themselves are never deserialized.
    """
    try:
        size = converted.stat().st_size
    except OSError:
        return False
    if size < 8:
        return False
    try:
        with converted.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                return False
            header_len = int.from_bytes(raw_len, "little")
            if header_len <= 0 or header_len > size - 8 or header_len > MAX_SAFETENSORS_HEADER:
                return False
            header = handle.read(header_len)
            if len(header) != header_len:
                return False
    except OSError:
        return False
    try:
        parsed = json.loads(header)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict) or not parsed:
        return False
    data_end = size - 8 - header_len
    tensors = 0
    for key, entry in parsed.items():
        if key == "__metadata__":
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] > offsets[1]
            or offsets[1] > data_end
        ):
            return False
        tensors += 1
    return tensors > 0


def refuse_hub_download(*, backend: str, route: str, size: str, detail: str = "") -> None:
    """Exit. The message names the backend, the route and the size."""
    message = (
        f"{backend} ({route}) needs its weights ({size}) on disk before a run. "
        "Generation does not fetch them."
    )
    if detail:
        message = f"{message}\n{detail}"
    raise SystemExit(message)


def hub_dir(hub: Path | None = None) -> Path:
    """The Hugging Face hub cache, honouring the usual variables."""
    if hub is not None:
        return hub
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def snapshot_dir(repo_id: str, hub: Path | None = None) -> Path | None:
    """A local snapshot of ``repo_id``, or None.

    A directory that already holds files is itself (a path the caller built).
    Otherwise the hub cache is searched. An empty folder is not a snapshot.
    """
    direct = Path(repo_id)
    if direct.is_dir() and any(direct.rglob("*")):
        return direct
    if "/" not in repo_id:
        return None
    model = hub_dir(hub) / f"models--{repo_id.replace('/', '--')}"
    snapshots = model / "snapshots"
    if not snapshots.is_dir():
        return None
    found = [snap for snap in snapshots.iterdir()
             if snap.is_dir() and any(snap.rglob("*"))]
    ref = model / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text().strip()
        current = snapshots / revision
        if current.parent == snapshots and current in found:
            return current
    return max(found, key=lambda snap: (snap.stat().st_mtime_ns, snap.name), default=None)


def require_local(repo_id: str, *, backend: str, route: str, size: str,
                  hub: Path | None = None) -> str:
    """The local snapshot path. Refuses when there is none.

    Also pins the process offline, so a later import cannot fetch the rest.
    """
    local = snapshot_dir(repo_id, hub)
    if local is None:
        refuse_hub_download(backend=backend, route=route, size=size)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return str(local)
