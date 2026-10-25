"""Importing a Mac patch must not edit a tree. Only main() may write."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

HIGHPOLY_GENERATE = (
    "def build():\n"
    "    parser.add_argument(\n"
    '        "--no-texture", action="store_true",\n'
    "    )\n"
    '    print(f"\\nMesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} triangles")\n'
)

QUALITY_GENERATE = """\
def build(parser):
        "--texture-size", type=int, default=1024,
        choices=[512, 1024, 2048],
    parser.add_argument(
        "--no-texture", action="store_true",
    )
                glb = o_voxel.postprocess.to_glb(
                    decimation_target=target_faces,
                    texture_size=tex_size,
                    verbose=True,
                )
"""


def _load_copy(root: Path, filename: str):
    scripts = root / "scripts"
    scripts.mkdir(exist_ok=True)
    destination = scripts / filename
    destination.write_bytes((SCRIPTS / filename).read_bytes())
    spec = importlib.util.spec_from_file_location(
        filename.replace(".py", "_copy"), destination
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(root: Path) -> dict[Path, bytes]:
    vendor = root / "vendor"
    if not vendor.exists():
        return {}
    return {path: path.read_bytes() for path in vendor.rglob("*") if path.is_file()}


def test_highpoly_import_does_not_write_until_main(tmp_path: Path):
    generate = tmp_path / "vendor" / "trellis-mac" / "generate.py"
    generate.parent.mkdir(parents=True)
    generate.write_text(HIGHPOLY_GENERATE)
    before = _snapshot(tmp_path)
    module = _load_copy(tmp_path, "patch_trellis_highpoly.py")
    assert _snapshot(tmp_path) == before
    assert module.main() == 0
    assert _snapshot(tmp_path) != before


def test_quality_import_does_not_write_until_main(tmp_path: Path):
    generate = tmp_path / "vendor" / "trellis-mac" / "generate.py"
    generate.parent.mkdir(parents=True)
    generate.write_text(QUALITY_GENERATE)
    before = _snapshot(tmp_path)
    module = _load_copy(tmp_path, "patch_trellis_quality.py")
    assert _snapshot(tmp_path) == before
    assert module.main() == 0
    assert generate.read_text() != QUALITY_GENERATE


def test_highpoly_import_with_no_tree_does_not_raise(tmp_path: Path):
    module = _load_copy(tmp_path, "patch_trellis_highpoly.py")
    assert callable(module.main)


def test_quality_import_with_no_tree_does_not_raise(tmp_path: Path):
    module = _load_copy(tmp_path, "patch_trellis_quality.py")
    assert callable(module.main)
