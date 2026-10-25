"""The 2.1 converter uses the paths it is given and does not unpickle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "hunyuan_mlx" / "shape" / "scripts" / "convert_v21_ckpt.py"
DW = REPO / "hunyuan_mlx" / "download_weights.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


convert = _load("convert_v21_ckpt", SCRIPT)
dw = _load("download_weights_for_convert", DW)


class _Tensor:
    def is_floating_point(self):
        return True

    def half(self):
        return self

    def contiguous(self):
        return self


def test_convert_reads_the_two_paths_and_safe_load_only(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.fp16.ckpt"
    out = tmp_path / "nested" / "model.fp16.safetensors"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    calls = []

    def load(path, map_location=None, weights_only=None):
        calls.append((str(path), map_location, weights_only))
        if weights_only is not True:
            raise AssertionError("unsafe unpickle")
        return {
            "model": {"block.weight": _Tensor()},
            "vae": {"block.weight": _Tensor()},
            "conditioner": {"block.weight": _Tensor()},
        }

    saved = {}

    def save_file(tensors, path):
        saved["path"] = str(path)
        saved["keys"] = set(tensors)

    fake_torch = type(sys)("torch")
    fake_torch.load = load
    fake_torch.is_tensor = lambda value: isinstance(value, _Tensor)
    fake_st = type(sys)("safetensors.torch")
    fake_st.save_file = save_file
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "safetensors", type(sys)("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st)

    assert convert.main([str(ckpt), str(out)]) == 0
    assert calls == [(str(ckpt), "cpu", True)]
    assert saved["path"] == str(out)
    assert saved["keys"] == {
        "model.block.weight",
        "vae.block.weight",
        "conditioner.block.weight",
    }


def test_a_failed_safe_load_is_not_retried_unsafe(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.fp16.ckpt"
    ckpt.write_bytes(b"x")
    calls = []

    def load(path, map_location=None, weights_only=None):
        calls.append(weights_only)
        raise RuntimeError("weights_only failed")

    fake_torch = type(sys)("torch")
    fake_torch.load = load
    fake_torch.is_tensor = lambda value: False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="weights_only failed"):
        convert.convert(str(ckpt), str(tmp_path / "out.safetensors"))
    assert calls == [True]


def test_torch_python_is_the_oracle_venv_not_this_interpreter():
    paths = [str(path) for path in dw.torch_interpreters()]
    assert any(
        path.endswith(".venv-oracle/bin/python")
        or path.endswith(".venv-oracle\\Scripts\\python.exe")
        or path.endswith(".venv-oracle/Scripts/python.exe")
        for path in paths
    )
    assert sys.executable not in paths


def test_download_shape_runs_the_converter_with_those_arguments(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.fp16.ckpt"
    ckpt.write_bytes(b"ckpt")
    oracle = tmp_path / "oracle-python"
    oracle.write_text("oracle")
    seen = {}

    def hf_hub_download(_repo, _filename, **_kwargs):
        return str(ckpt)

    fake_hub = type(sys)("huggingface_hub")
    fake_hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(dw, "SHAPE_WEIGHTS", tmp_path / "weights")
    monkeypatch.setattr(dw, "torch_python", lambda: oracle)

    def run(cmd, cwd=None, check=True):
        seen["cmd"] = list(cmd)
        seen["cwd"] = cwd
        Path(cmd[-1]).write_bytes(b"")
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(dw.subprocess, "run", run)
    dw.download_shape("2.1")
    cmd = seen["cmd"]
    assert cmd[0] == str(oracle)
    assert cmd[0] != sys.executable
    assert cmd[1].endswith("convert_v21_ckpt.py")
    assert cmd[2] == str(ckpt)
    assert cmd[3].endswith("model.fp16.safetensors")
    assert Path(seen["cwd"]) == dw.REPO / "shape"
    assert ckpt.is_file()


def test_the_checkpoint_goes_once_the_safetensors_has_the_shape_tensors(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    ckpt = tmp_path / "model.fp16.ckpt"
    converted = tmp_path / "model.fp16.safetensors"
    ckpt.write_bytes(b"\0" * 64)
    save_file(
        {
            "conditioner.w": np.zeros((1,), np.float16),
            "model.w": np.zeros((1,), np.float16),
            "vae.w": np.zeros((1,), np.float16),
        },
        converted,
    )
    assert dw.discard_converted_checkpoint(ckpt, converted) is True
    assert not ckpt.exists()


def test_a_safetensors_missing_a_prefix_keeps_the_checkpoint(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    ckpt = tmp_path / "model.fp16.ckpt"
    converted = tmp_path / "model.fp16.safetensors"
    ckpt.write_bytes(b"\0" * 64)
    save_file({"model.w": np.zeros((1,), np.float16)}, converted)
    assert dw.discard_converted_checkpoint(ckpt, converted) is False
    assert ckpt.is_file()


def test_a_stub_safetensors_does_not_delete_the_checkpoint(tmp_path):
    ckpt = tmp_path / "model.fp16.ckpt"
    converted = tmp_path / "model.fp16.safetensors"
    ckpt.write_bytes(b"\0" * 4096)
    converted.write_bytes(b"\0" * 4096)
    assert dw.discard_converted_checkpoint(ckpt, converted) is False
    assert ckpt.is_file()
