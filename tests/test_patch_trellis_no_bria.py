"""Mac BRIA patch: a constructor beside the old note must still be removed.

Importing the real script used to patch vendor/ on the way in, so these tests
run a copy whose project root is a temp folder. The real tree is never written.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "patch_trellis_no_bria.py"
FACE_CAP = REPO / "scripts" / "patch_trellis_face_cap.py"

NEEDLE = (
    "        pipeline.rembg_model = getattr(rembg, "
    "args['rembg_model']['name'])(**args['rembg_model']['args'])"
)
DISABLE = (
    "        # image-to-3dlab disables separately licensed background models.\n"
    "        # Inputs must already contain a non-opaque alpha channel.\n"
    "        pipeline.rembg_model = None"
)

GENERATE_OK = """\
def build():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-texture", action="store_true",
    )

def bake(args, faces, faces_np):
            target_faces = min(200000, len(faces))
                target_faces = min(200000, len(faces_np))
"""

GENERATE_SHARED = """\
def build():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-texture", action="store_true",
    )

def bake(args, faces, faces_np):
            target_faces = min(args.bake_target_faces, len(faces))
                target_faces = min(args.bake_target_faces, 200000, len(faces_np))
"""


def _copy_script(root: Path) -> Path:
    scripts = root / "scripts"
    scripts.mkdir(exist_ok=True)
    destination = scripts / "patch_trellis_no_bria.py"
    destination.write_bytes(SCRIPT.read_bytes())
    return destination


def _pipelines(root: Path) -> Path:
    path = root / "vendor" / "trellis-mac" / "TRELLIS.2" / "trellis2" / "pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _generate(root: Path) -> Path:
    path = root / "vendor" / "trellis-mac" / "generate.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_pipelines(root: Path, text: str) -> list[Path]:
    folder = _pipelines(root)
    paths = []
    for name in ("trellis2_image_to_3d.py", "trellis2_texturing.py"):
        path = folder / name
        path.write_text(text)
        paths.append(path)
    return paths


def _load_copy(root: Path):
    destination = _copy_script(root)
    spec = importlib.util.spec_from_file_location(
        "patch_trellis_no_bria_copy", destination
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_copy(root: Path) -> subprocess.CompletedProcess[str]:
    script = _copy_script(root)
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_with_the_vendor_tree_absent_does_not_raise(tmp_path: Path):
    module = _load_copy(tmp_path)
    assert callable(module.main)


def test_a_constructor_beside_the_disable_text_is_removed_before_success(
    tmp_path: Path,
):
    """The comment is not proof. Success is only allowed once the constructor is gone."""
    dirty = DISABLE + "\n" + NEEDLE + "\n"
    pipelines = _write_pipelines(tmp_path, dirty)
    _generate(tmp_path).write_text(GENERATE_OK)

    result = _run_copy(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "BRIA will not load" in result.stdout
    for path in pipelines:
        text = path.read_text()
        assert NEEDLE not in text
        assert "pipeline.rembg_model = None" in text
    # A second run does not rewrite a file that is already clean.
    before = [path.read_text() for path in pipelines]
    again = _run_copy(tmp_path)
    assert again.returncode == 0, again.stderr
    assert [path.read_text() for path in pipelines] == before


def test_missing_background_hook_is_loud(tmp_path: Path):
    pipelines = _write_pipelines(tmp_path, "pipeline = object()\n")
    _generate(tmp_path).write_text(GENERATE_OK)
    module = _load_copy(tmp_path)
    with pytest.raises(RuntimeError, match="background-model hook"):
        module.patch(pipelines[0])


def test_face_cap_then_bria_keeps_the_raised_ceiling(tmp_path: Path):
    """The shared budget line is any integer ceiling, not the digits 200000."""
    pipelines = _write_pipelines(tmp_path, DISABLE + "\n")
    generate = _generate(tmp_path)
    generate.write_text(GENERATE_SHARED)
    capped = subprocess.run(
        [
            sys.executable,
            str(FACE_CAP),
            "--target",
            str(generate),
            "--ceiling",
            "300000",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert capped.returncode == 0, capped.stderr

    result = _run_copy(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "BRIA will not load" in result.stdout
    text = generate.read_text()
    assert "min(args.bake_target_faces, 200000, len(faces_np))" not in text
    assert "min(args.bake_target_faces, 300000, len(faces_np))" in text
    for path in pipelines:
        assert NEEDLE not in path.read_text()
    again = _run_copy(tmp_path)
    assert again.returncode == 0, again.stderr
    assert generate.read_text() == text
