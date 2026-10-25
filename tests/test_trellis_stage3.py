"""Pure contract tests for the Stage-3-only TRELLIS runner."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trellis_stage3.py"
SPEC = importlib.util.spec_from_file_location("trellis_stage3", SCRIPT)
stage3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage3)


def _bundle():
    return {
        "shape_slat_feats": torch.zeros((7, 32)),
        "coords": torch.zeros((7, 4), dtype=torch.int32),
        "res": 1024,
        "pipeline_type": "1024_cascade",
        "images": ["source.png"],
    }


def test_valid_bundle_is_accepted():
    stage3.validate_bundle(_bundle())


@pytest.mark.parametrize(
    "resolution,expected",
    [
        (512, (512, "tex_slat_flow_model_512")),
        (1024, (1024, "tex_slat_flow_model_1024")),
        (1536, (1024, "tex_slat_flow_model_1024")),
    ],
)
def test_material_stage_contract_selects_matching_model(resolution, expected):
    payload = _bundle()
    payload["res"] = resolution
    assert stage3.material_stage_contract(payload) == expected


class _Voxel:
    feats = torch.ones((3, 6))
    coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]])


def test_material_payload_supports_clean_space_decode_schema():
    geometry = {
        "vertices": "V",
        "faces": "F",
        "attrs": "old attrs",
        "coords": "old coords",
        "attr_layout": {"base_color": slice(0, 3)},
        "res": 512,
    }
    bundle = _bundle()
    bundle["res"] = 512
    payload = stage3.build_material_payload(
        geometry, _Voxel(), bundle, 42, {"steps": 12}
    )
    assert payload["vertices"] == "V"
    assert payload["faces"] == "F"
    assert payload["attrs"].shape == (3, 6)
    assert payload["coords"].shape == (3, 3)
    assert payload["texture_seed"] == 42
    assert payload["res"] == 512


def test_material_payload_preserves_legacy_split_cache_schema():
    geometry = {
        "geometry_ref": "geometry.pt",
        "layout": {"base_color": slice(0, 3)},
        "voxel_size": 0.5,
    }
    payload = stage3.build_material_payload(
        geometry, _Voxel(), _bundle(), 7, {"steps": 12}
    )
    assert payload["geometry_ref"] == "geometry.pt"
    assert payload["voxel_size"] == 0.5
    assert payload["texture_seed"] == 7


def test_missing_shape_latent_is_rejected():
    payload = _bundle()
    del payload["shape_slat_feats"]
    with pytest.raises(ValueError, match="shape_slat_feats"):
        stage3.validate_bundle(payload)


def test_row_mismatch_is_rejected():
    payload = _bundle()
    payload["coords"] = torch.zeros((6, 4), dtype=torch.int32)
    with pytest.raises(ValueError, match="row mismatch"):
        stage3.validate_bundle(payload)


def test_default_geometry_path_matches_generate_outputs(tmp_path):
    path = tmp_path / "snag_latents.pt"
    assert stage3.default_geometry_decode(path) == tmp_path / "snag_decode.pt"


def test_official_texture_defaults_are_explicit():
    args = argparse.Namespace(
        steps=12,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        guidance_interval=(0.6, 0.9),
        rescale_t=3.0,
    )
    assert stage3.sampler_params(args) == {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "guidance_interval": (0.6, 0.9),
        "rescale_t": 3.0,
    }


def test_runner_can_target_the_clean_space_port_and_stop_after_sampling():
    source = SCRIPT.read_text()
    assert '"--vendor-root"' in source
    assert 'default="sdpa"' in source
    assert '"metal_flash"' in source
    assert '"--sample-only"' in source
    assert '"--preprocessed-image"' in source
    assert "material decode intentionally skipped" in source
    assert "load_rembg=False" in source
    assert "load_rembg=True" not in source
    assert "load_rembg=not has_transparent_alpha" not in source


def _isolate_weights(monkeypatch, tmp_path):
    import dataclasses

    import trellis_route

    monkeypatch.delenv("I2L_TRELLIS_MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(
        trellis_route,
        "CUDA",
        dataclasses.replace(trellis_route.CUDA, vendor=tmp_path / "trellis-cuda"),
    )
    monkeypatch.setattr(
        trellis_route,
        "ROCM",
        dataclasses.replace(trellis_route.ROCM, vendor=tmp_path / "trellis-rocm"),
    )


def test_an_opaque_picture_is_refused_unless_it_was_preprocessed(tmp_path):
    from PIL import Image

    image = tmp_path / "plain.png"
    Image.new("RGB", (8, 8), "white").save(image)
    with pytest.raises(SystemExit, match="transparent"):
        stage3.refuse_uncut_image(image, preprocessed=False)
    stage3.refuse_uncut_image(image, preprocessed=True)


def test_a_live_remover_constructor_is_refused(tmp_path):
    import patch_trellis_space_core as space_patch

    pipeline = (
        tmp_path / "TRELLIS.2" / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
    )
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text("rembg_model = None\n")
    assert stage3.live_remover_refusal(tmp_path) is None
    pipeline.write_text(space_patch._CONSTRUCTOR + "\n")
    text = stage3.live_remover_refusal(tmp_path)
    assert text and "BRIA" in text and "patch_trellis_space_core.py" in text
    assert stage3.live_remover_refusal(tmp_path / "absent") is None


def test_open_pipeline_does_not_load_when_the_tree_is_missing(monkeypatch, tmp_path):
    _isolate_weights(monkeypatch, tmp_path)
    monkeypatch.setattr(stage3, "load_trellis", lambda model: pytest.fail(model))
    with pytest.raises(SystemExit, match="GB") as refused:
        stage3.open_pipeline(tmp_path / "vendor", confirm=False)
    assert "TRELLIS.2" in str(refused.value)
    assert "--yes" in str(refused.value)
