"""Tests for the Pixal3D BRIA substitution.

This is a licence guardrail, not a preference, so the properties that matter are that it
fires on the exact string the shipped weights ask for, that it cannot be applied twice,
and that it fails loudly when the anchor moves — a silently skipped patch here means a run
downloads BRIA RMBG-2.0.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_pixal3d_rembg.py"


def _load():
    spec = importlib.util.spec_from_file_location("patch_pixal3d_rembg", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

# The upstream file as it ships, trimmed to what the patch touches.
ORIGINAL = """from typing import *
from transformers import AutoModelForImageSegmentation


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
"""


def test_the_exact_name_the_shipped_weights_ask_for_is_substituted():
    """pipeline.json in TencentARC/Pixal3D carries this literal string."""
    assert patch.permissive_model_name("briaai/RMBG-2.0") == "ZhengPeng7/BiRefNet"


def test_any_bria_model_is_substituted_not_just_that_one():
    assert patch.permissive_model_name("briaai/RMBG-1.4") == "ZhengPeng7/BiRefNet"


def test_a_permissive_model_is_passed_through_untouched():
    assert patch.permissive_model_name("ZhengPeng7/BiRefNet") == "ZhengPeng7/BiRefNet"
    assert patch.permissive_model_name("some/other-matting") == "some/other-matting"


def test_applying_the_patch_inserts_the_guard():
    patched = patch.apply(ORIGINAL)
    assert patch.is_patched(patched)
    assert "briaai/" in patched
    assert "ZhengPeng7/BiRefNet" in patched
    # The guard must sit inside __init__, before the model is fetched.
    assert patched.index(patch.MARKER) < patched.index("from_pretrained")


def test_the_patch_is_idempotent():
    once = patch.apply(ORIGINAL)
    twice = patch.apply(once)
    assert once == twice
    assert once.count(patch.MARKER) == twice.count(patch.MARKER)


def test_a_missing_anchor_fails_loudly():
    """Silently doing nothing here would mean a run loads BRIA."""
    moved = ORIGINAL.replace(
        'def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):',
        "def __init__(self, model_name=None, **kwargs):",
    )
    with pytest.raises(SystemExit):
        patch.apply(moved)


def test_a_marker_without_the_guard_is_not_patched_and_a_second_apply_puts_it_back(
    tmp_path,
):
    broken = ORIGINAL.replace(
        '    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):\n',
        '    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):\n'
        f"        # {patch.MARKER}: note without the guard\n",
    )
    assert patch.MARKER in broken
    assert 'model_name.startswith("briaai/")' not in broken
    path = tmp_path / "BiRefNet.py"
    path.write_text(broken)

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(path), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    checked = run("--check")
    assert checked.returncode != 0
    applied = run()
    assert applied.returncode == 0, applied.stderr
    restored = path.read_text()
    assert 'model_name.startswith("briaai/")' in restored
    assert patch.is_patched(restored)
    again = run()
    assert again.returncode == 0, again.stderr
    assert path.read_text() == restored


def test_a_bria_constructor_argument_is_not_treated_as_patched(tmp_path):
    bad = patch.apply(ORIGINAL)
    bad += (
        '\nmodel = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0")\n'
    )
    path = tmp_path / "BiRefNet.py"
    path.write_text(bad)
    assert patch.is_patched(bad) is False
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    applied = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert applied.returncode != 0
    assert "briaai/RMBG-2.0" in path.read_text()


def test_the_injected_guard_is_valid_python_and_behaves():
    """Compile the patched file and drive the guard, rather than trusting the string."""
    patched = patch.apply(ORIGINAL)
    # Keep only the injected branch, with the model load stubbed out.
    body = patched.replace(
        """        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
""",
        "        self.model_name = model_name\n",
    ).replace("from transformers import AutoModelForImageSegmentation\n", "")

    namespace: dict = {}
    exec(compile(body, "patched_BiRefNet.py", "exec"), namespace)

    assert namespace["BiRefNet"]("briaai/RMBG-2.0").model_name == "ZhengPeng7/BiRefNet"
    assert (
        namespace["BiRefNet"]("ZhengPeng7/BiRefNet").model_name == "ZhengPeng7/BiRefNet"
    )
