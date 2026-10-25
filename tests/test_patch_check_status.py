"""`--check` is an error when the patch is not in place. A missing tree stays 2."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import patch_metal_hashmap_miss as hashmap  # noqa: E402
import patch_mtlbvh_production_traversal as mtlbvh  # noqa: E402
import patch_ovoxel_remesh_checkpoints as remesh  # noqa: E402
import patch_trellis_enable_cleanup as cleanup  # noqa: E402
import patch_trellis_face_cap as face_cap  # noqa: E402
import patch_trellis_mlx_attention as mlx  # noqa: E402


def _argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", list(args))


def test_face_cap_check_on_unpatched_source_exits_1(tmp_path: Path, monkeypatch):
    path = tmp_path / "generate.py"
    path.write_text(
        "target_faces = min(args.bake_target_faces, 200000, len(faces_np))\n"
    )
    _argv(monkeypatch, "face-cap", "--check", "--target", str(path))
    assert face_cap.main() == 1
    path.write_text(face_cap.apply(path.read_text(), 300_000))
    assert face_cap.main() == 0


def test_cleanup_check_on_unpatched_source_exits_1(tmp_path: Path, monkeypatch):
    path = tmp_path / "base.py"
    path.write_text("".join(signature + stub for signature, stub in cleanup.STUBS))
    _argv(monkeypatch, "cleanup", "--check", "--path", str(path))
    assert cleanup.main() == 1
    enabled, _changed = cleanup.enable(path.read_text())
    path.write_text(enabled)
    assert cleanup.main() == 0


def test_hashmap_check_on_unpatched_source_exits_1(tmp_path: Path, monkeypatch):
    path = tmp_path / "remesh.metal"
    path.write_text(
        "".join(
            signature + "\n" + original
            for signature, original, _patched in hashmap.PATCHES
        )
    )
    _argv(monkeypatch, "hashmap", "--check", "--path", str(path))
    assert hashmap.main() == 1
    path.write_text(hashmap.apply(path.read_text()))
    assert hashmap.main() == 0
    _argv(monkeypatch, "hashmap", "--check", "--path", str(tmp_path / "missing.metal"))
    assert hashmap.main() == 2


def test_remesh_checkpoint_check_on_unpatched_source_exits_1(
    tmp_path: Path, monkeypatch
):
    absent = tmp_path / "postprocess.py"
    absent.write_text("not patched\n")
    missing = tmp_path / "missing.py"
    _argv(monkeypatch, "remesh", "--check", "--target", str(missing))
    assert remesh.main() == 2
    _argv(monkeypatch, "remesh", "--check", "--target", str(absent))
    assert remesh.main() == 1
    present = tmp_path / "patched.py"
    present.write_text(remesh.MARKER + "\n")
    _argv(monkeypatch, "remesh", "--check", "--target", str(present))
    assert remesh.main() == 0


def test_mlx_check_on_unpatched_source_exits_1(tmp_path: Path, monkeypatch):
    root = tmp_path / "TRELLIS.2"
    config = root / "trellis2" / "modules" / "sparse" / "config.py"
    attn = root / "trellis2" / "modules" / "sparse" / "attention" / "full_attn.py"
    config.parent.mkdir(parents=True)
    attn.parent.mkdir(parents=True)
    config.write_text(mlx.BACKEND_LIST_OLD + "\n" + mlx.BACKEND_LIST_OLD + "\n")
    attn.write_text(mlx.DISPATCH_ANCHOR + "\n")
    before = config.read_text(), attn.read_text()
    _argv(monkeypatch, "mlx", "--check", "--root", str(root))
    assert mlx.main() == 1
    assert (config.read_text(), attn.read_text()) == before
    mlx.main(["--root", str(root)])
    assert mlx.main(["--check", "--root", str(root)]) == 0


def test_mtlbvh_check_on_unpatched_source_exits_1(tmp_path: Path, monkeypatch):
    metal = tmp_path / "bvh.metal"
    native = tmp_path / "metal_bvh.mm"
    metal.write_text("stack based traversal only\n")
    native.write_text(mtlbvh.CPU_OLD + "\n")
    monkeypatch.setattr(mtlbvh, "METAL", metal)
    monkeypatch.setattr(mtlbvh, "NATIVE", native)
    _argv(monkeypatch, "mtlbvh", "--check")
    assert mtlbvh.main() == 1
    metal.write_text("int closest_triangle_stackless(point)\n")
    native.write_text(mtlbvh.CPU_NEW)
    assert mtlbvh.main() == 0
    monkeypatch.setattr(mtlbvh, "METAL", tmp_path / "gone.metal")
    monkeypatch.setattr(mtlbvh, "NATIVE", tmp_path / "gone.mm")
    assert mtlbvh.main() == 2
