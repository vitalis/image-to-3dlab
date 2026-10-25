"""The CUDA probe must not download TRELLIS weights when pipeline.json is missing."""

from __future__ import annotations

import dataclasses

import pytest
import runpod_trellis2_cuda_probe as probe
import trellis_route


def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("I2L_TRELLIS_MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(
        trellis_route,
        "CUDA",
        dataclasses.replace(trellis_route.CUDA, vendor=tmp_path / "c"),
    )
    monkeypatch.setattr(
        trellis_route,
        "ROCM",
        dataclasses.replace(trellis_route.ROCM, vendor=tmp_path / "r"),
    )


def test_a_missing_tree_does_not_load_the_pipeline(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(probe, "load_pretrained", lambda model: pytest.fail(model))
    with pytest.raises(SystemExit, match="GB") as refused:
        probe.begin(False)
    text = str(refused.value)
    assert "TRELLIS.2" in text
    assert "--yes" in text


def test_yes_is_the_only_way_the_hub_id_is_handed_over(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(
        probe, "load_pretrained", lambda model: seen.append(model) or object()
    )
    probe.begin(True)
    assert seen == [trellis_route.HUB_REPO]
