"""Contracts for the official-Space-first Mac compatibility patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_trellis_space_core.py"
SPEC = importlib.util.spec_from_file_location("patch_trellis_space_core", SCRIPT)
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)


def test_replacement_fails_closed_when_upstream_anchor_changes():
    try:
        patcher.replace_exact("upstream changed", "old", "new", count=1, label="gate")
    except RuntimeError as error:
        assert "expected 1 source anchors, found 0" in str(error)
    else:
        raise AssertionError("an unknown upstream shape must not be patched silently")


def test_replacement_is_idempotent():
    source, changed = patcher.replace_exact("old", "old", "new", count=1, label="gate")
    assert (source, changed) == ("new", True)
    source, changed = patcher.replace_exact(source, "old", "new", count=1, label="gate")
    assert (source, changed) == ("new", False)


def test_sparse_attention_patch_uses_packed_metal_kernel():
    source = SCRIPT.read_text()
    assert "flex_gemm.kernels.metal.sparse_attention_fwd" in source
    assert "q_padded" not in source
    assert "kv_prefix" in source


def test_sdpa_preserves_packed_sequence_boundaries_and_metal_fails_fast_at_128():
    source = SCRIPT.read_text()
    assert "for q_length, kv_length in zip(q_seqlen, kv_seqlen)" in source
    assert "F.scaled_dot_product_attention" in source
    assert "if q.shape[-1] > 64" in source
    assert "Use the sdpa backend for TRELLIS.2-4B" in source


def test_core_patch_does_not_change_model_or_sampler_equations():
    source = SCRIPT.read_text()
    forbidden_targets = (
        "samplers/",
        "models/sparse_structure_flow.py",
        "models/structured_latent_flow.py",
        "tex_slat_normalization",
        "shape_slat_normalization",
        "preprocess_image(self",
    )
    for target in forbidden_targets:
        assert target not in source


UPSTREAM_PIPELINE = """\
class _Rembg:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def build(**kwargs):
            self.calls.append((name, kwargs))
            return kwargs
        return build


rembg = _Rembg()


class Trellis2ImageTo3DPipeline:
    def from_pretrained(path: str) -> "Trellis2ImageTo3DPipeline":
        args = {"rembg_model": {"name": "BiRefNet", "args": {"model_name": "briaai/RMBG-2.0"}}}
        new_pipeline = Trellis2ImageTo3DPipeline()
        new_pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        return new_pipeline
"""


def _patched_pipeline(tmp_path: Path):
    root = tmp_path / "space"
    path = root / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
    path.parent.mkdir(parents=True)
    path.write_text(UPSTREAM_PIPELINE)
    assert patcher.patch_optional_rembg(root) is True
    spec = importlib.util.spec_from_file_location("patched_space_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def test_from_pretrained_does_not_request_bria(tmp_path: Path):
    """No keyword, and an explicit true, must not construct briaai/RMBG-2.0."""
    path, module = _patched_pipeline(tmp_path)
    pipeline = module.Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B"
    )
    assert pipeline.rembg_model is None
    assert module.rembg.calls == []
    with pytest.raises(RuntimeError, match="BRIA"):
        module.Trellis2ImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B", load_rembg=True
        )
    assert module.rembg.calls == []
    text = path.read_text()
    assert "getattr(rembg, args['rembg_model']['name'])" not in text
    again = path.read_text()
    assert patcher.patch_optional_rembg(path.parents[2]) is False
    assert path.read_text() == again
