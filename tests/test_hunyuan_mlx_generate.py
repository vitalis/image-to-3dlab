"""dgrauet Hunyuan MLX: local weights, a licence file, and a paint folder per run."""

from __future__ import annotations

from pathlib import Path

import hunyuan_mlx_generate as gen
import pytest


def test_missing_weights_do_not_call_from_pretrained(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    empty = hub / "models--dgrauet--hunyuan3d-2.1-mlx" / "snapshots" / "abc"
    empty.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        gen, "_shape_from_pretrained", lambda *a, **k: pytest.fail("loaded")
    )
    with pytest.raises(SystemExit, match="from_pretrained") as refused:
        gen.run_shape(tmp_path / "in.png", 256, 1, 0.0)
    text = str(refused.value)
    assert "Hunyuan3D-MLX" in text
    assert "dgrauet/hunyuan3d-2.1-mlx" in text
    assert "13.0 GB" in text
    assert "bootstrap_hunyuan_xiong.py" in text
    assert "docs/hunyuan-mlx-recipes.md" in text


def test_a_local_snapshot_is_what_gets_loaded(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    snap = hub / "models--dgrauet--hunyuan3d-2.1-mlx" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    seen = []

    class _Pipe:
        def __call__(self, *_args, **_kwargs):
            return type("Mesh", (), {"vertices": [1], "faces": [1]})()

    monkeypatch.setattr(
        gen,
        "_shape_from_pretrained",
        lambda weights: seen.append(Path(weights)) or _Pipe(),
    )
    gen.run_shape(tmp_path / "in.png", 256, 1, 0.0)
    assert seen == [snap]


def test_two_outputs_get_two_paint_folders(tmp_path):
    one = gen.paint_work_dir(tmp_path / "one.glb")
    two = gen.paint_work_dir(tmp_path / "two.glb")
    assert one != two
    assert one.name == "one__paint"
    env = gen.paint_env(
        tmp_path / "mesh.obj",
        tmp_path / "in.png",
        tmp_path / "one.glb",
        0,
        512,
        15,
        4096,
    )
    assert env["PAINT_OUT"] == str(one.resolve())
    assert "textured_mesh_pbr.glb" not in env["PAINT_OUT"]


def test_provenance_records_the_territorial_licence(tmp_path):
    image = tmp_path / "in.png"
    output = tmp_path / "out.glb"
    image.write_bytes(b"png")
    output.write_bytes(b"glb")
    record = gen.build_provenance(image=image, output=output, parameters={"seed": 1})
    assert record["license"]["name"]
    assert any("EU" in condition for condition in record["license"]["conditions"])
    assert record["model"] == {
        "backend": "hunyuan-mlx",
        "route": "apple-mlx",
        "parameters": {"seed": 1},
    }
    names = [item["component"] for item in record["components"]]
    assert names == ["facebook/dinov2-giant", "RealESRGAN_x4plus"]
    assert gen.provenance_path(output) == output.with_suffix(".provenance.json")


def test_octree_script_requires_an_image_and_an_output(tmp_path, monkeypatch):
    import hunyuan_shape_octree_test as octree

    source = Path(octree.__file__).read_text()
    assert "game-controller-test.png" not in source
    assert "hunyuan_mlx_zimeng_test" not in source
    with pytest.raises(SystemExit):
        octree.main([])
    image = tmp_path / "pic.png"
    image.write_bytes(b"png")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    with pytest.raises(SystemExit, match="from_pretrained"):
        octree.main(["256", str(image), str(tmp_path / "out")])
