"""Weight downloads wait for a yes. Nothing here touches the network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dw = _load("download_weights", REPO / "hunyuan_mlx" / "download_weights.py")
any_dl = _load("dl_any", REPO / "hunyuan_mlx" / "shape" / "scripts" / "dl_any.py")
ms_dl = _load(
    "dl_modelscope", REPO / "hunyuan_mlx" / "shape" / "scripts" / "dl_modelscope.py"
)
esr = _load(
    "convert_realesrgan",
    REPO / "hunyuan_mlx" / "paint" / "scripts" / "convert_realesrgan.py",
)


def _refuse_network(monkeypatch, module, names):
    def boom(*_args, **_kwargs):
        raise AssertionError("downloaded")

    for name in names:
        monkeypatch.setattr(module, name, boom, raising=False)


def test_main_returns_before_any_hub_download_without_yes(monkeypatch, capsys):
    monkeypatch.setattr(dw.sys.stdin, "isatty", lambda: False)
    _refuse_network(monkeypatch, dw, ["download_shape", "download_paint"])
    assert dw.main([]) == 1
    out = capsys.readouterr().out
    assert "EU" in out and "South Korea" in out
    assert "nobody to ask" in out


def test_eof_on_a_tty_does_not_download(monkeypatch, capsys):
    monkeypatch.setattr(dw.sys.stdin, "isatty", lambda: True)

    def closed(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)
    _refuse_network(monkeypatch, dw, ["download_shape", "download_paint"])
    assert dw.main([]) == 1
    assert "nobody to ask" in capsys.readouterr().out


def test_modelscope_any_returns_before_the_file_without_yes(monkeypatch, capsys):
    monkeypatch.setattr(any_dl.sys.stdin, "isatty", lambda: False)
    _refuse_network(monkeypatch, any_dl, ["download_file", "lookup_size"])
    assert any_dl.main(["Tencent-Hunyuan/Hunyuan3D-2", "a.safetensors", "out"]) == 1
    out = capsys.readouterr().out
    assert "EU" in out and "South Korea" in out
    assert "nobody to ask" in out


def test_modelscope_mini_returns_before_the_file_without_yes(monkeypatch, capsys):
    monkeypatch.setattr(ms_dl.sys.stdin, "isatty", lambda: False)
    _refuse_network(monkeypatch, ms_dl, ["download"])
    assert ms_dl.main([]) == 1
    out = capsys.readouterr().out
    assert "EU" in out and "South Korea" in out
    assert "nobody to ask" in out


def test_realesrgan_fetch_returns_before_the_file_without_yes(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(esr.sys.stdin, "isatty", lambda: False)

    def boom(*_args, **_kwargs):
        raise AssertionError("downloaded")

    monkeypatch.setattr(esr.urllib.request, "urlretrieve", boom)
    assert esr.main(["--out", str(tmp_path / "rrdbnet.npz")]) == 1
    out = capsys.readouterr().out
    assert "EU" in out and "South Korea" in out
    assert "nobody to ask" in out
    assert not (tmp_path / "RealESRGAN_x4plus.pth").exists()


def test_shape_fetch_does_not_invent_a_revision(monkeypatch, tmp_path):
    """No Hugging Face commit for the shape repos is written down here.

    Paint is pinned. The CUDA route pins a GitHub commit of Tencent's code,
    which is not a snapshot of tencent/Hunyuan3D-2 or tencent/Hunyuan3D-2.1.
    """
    seen = []

    def hf_hub_download(repo, filename, **kwargs):
        seen.append(kwargs)
        return str(tmp_path / "file")

    fake = type(sys)("huggingface_hub")
    fake.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setattr(dw, "SHAPE_WEIGHTS", tmp_path / "weights")
    dw.download_shape("2.0")
    assert seen, "the shape files were not requested"
    assert all(kwargs.get("revision") in (None, "") for kwargs in seen)
