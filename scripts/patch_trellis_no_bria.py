#!/usr/bin/env python3
"""Disable TRELLIS' configured background model for license-controlled runs.

Safe to run twice. A disable note left beside a live constructor is not
"already patched": the constructor is removed, and success is printed only
when that line is gone from both pipeline files.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ROOT / "vendor" / "trellis-mac" / "TRELLIS.2" / "trellis2" / "pipelines"
PIPELINE_NAMES = ("trellis2_image_to_3d.py", "trellis2_texturing.py")

NEEDLE = (
    "        pipeline.rembg_model = getattr(rembg, "
    "args['rembg_model']['name'])(**args['rembg_model']['args'])"
)
REPLACEMENT = (
    "        # image-to-3dlab disables separately licensed background models.\n"
    "        # Inputs must already contain a non-opaque alpha channel.\n"
    "        pipeline.rembg_model = None"
)

# Any integer ceiling counts. The face-cap patch rewrites 200000, and this
# script must not put that number back.
METAL_CAPPED = re.compile(
    r"min\(\s*args\.bake_target_faces\s*,\s*\d+\s*,\s*len\(faces_np\)\s*\)"
)


def patch(path: Path) -> None:
    source = path.read_text()
    if NEEDLE in source:
        updated = source.replace(NEEDLE, "" if REPLACEMENT in source else REPLACEMENT)
        if NEEDLE in updated:
            raise RuntimeError(f"BRIA constructor still present in {path}")
    elif "pipeline.rembg_model = None" not in source:
        raise RuntimeError(
            f"expected TRELLIS background-model hook not found in {path}"
        )
    else:
        updated = source
    if updated != source:
        path.write_text(updated)


def patch_cpu_bake_budget(path: Path) -> None:
    source = path.read_text()
    option_anchor = (
        '    parser.add_argument(\n        "--no-texture", action="store_true",\n'
    )
    option = (
        "    parser.add_argument(\n"
        '        "--bake-target-faces", type=int, default=50000,\n'
        '        help="Triangle budget for the CPU UV/texture fallback",\n'
        "    )\n"
    )
    if '"--bake-target-faces"' not in source:
        if option_anchor not in source:
            raise RuntimeError(
                f"expected TRELLIS CLI option anchor not found in {path}"
            )
        source = source.replace(option_anchor, option + option_anchor)

    budget_needle = "            target_faces = min(200000, len(faces))"
    budget_replacement = (
        "            target_faces = min(args.bake_target_faces, len(faces))"
    )
    if budget_needle in source:
        source = source.replace(budget_needle, budget_replacement)
    elif budget_replacement not in source:
        raise RuntimeError(f"expected TRELLIS bake budget hook not found in {path}")

    # The Metal bake path has its own budget line reading `faces_np`. The
    # 200,000 is a crash guard, so a first apply keeps it as a ceiling. A later
    # face-cap ceiling is already the hook and is left alone.
    metal_needle = "                target_faces = min(200000, len(faces_np))"
    metal_replacement = (
        "                target_faces = min("
        "args.bake_target_faces, 200000, len(faces_np))"
    )
    if metal_needle in source:
        source = source.replace(metal_needle, metal_replacement)
    elif METAL_CAPPED.search(source) is None:
        raise RuntimeError(
            f"expected TRELLIS Metal bake budget hook not found in {path}"
        )
    if source != path.read_text():
        path.write_text(source)


def _pipelines(root: Path) -> list[Path]:
    folder = root / "vendor" / "trellis-mac" / "TRELLIS.2" / "trellis2" / "pipelines"
    return [folder / name for name in PIPELINE_NAMES]


def _generate(root: Path) -> Path:
    return root / "vendor" / "trellis-mac" / "generate.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    pipelines = _pipelines(root)
    for path in pipelines:
        patch(path)
    patch_cpu_bake_budget(_generate(root))
    for path in pipelines:
        if NEEDLE in path.read_text():
            print(
                "BRIA constructor is still present; not taking this as success.",
                file=sys.stderr,
            )
            return 1
    print(
        "Patched TRELLIS for pre-matted RGBA inputs and a practical CPU bake budget; "
        "BRIA will not load."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
