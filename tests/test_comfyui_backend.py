from image_to_3dlab.comfyui_backend import ComfyUIClient, find_asset, patch_input_image


def test_patch_auto_detected_load_image():
    workflow = {"7": {"class_type": "LoadImage", "inputs": {"image": "old.png"}}}
    assert patch_input_image(workflow, "incoming/new.png", None) == "7"
    assert workflow["7"]["inputs"]["image"] == "incoming/new.png"


def test_find_nested_glb_output():
    entry = {
        "outputs": {
            "42": {
                "meshes": [
                    {"filename": "model.glb", "subfolder": "hunyuan", "type": "output"}
                ]
            }
        }
    }
    assert find_asset(entry)["filename"] == "model.glb"


def test_a_download_does_not_overwrite_a_file_already_there(tmp_path):
    existing = tmp_path / "model.glb"
    existing.write_bytes(b"keep-me")

    class Response:
        def raise_for_status(self):
            return None

        def iter_content(self, _size):
            yield b"new-bytes"

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    client = ComfyUIClient("http://127.0.0.1:8188", session=Session())
    saved = client._download(
        {"filename": "model.glb", "subfolder": "", "type": "output"}, tmp_path
    )
    assert existing.read_bytes() == b"keep-me"
    assert saved != existing and saved.read_bytes() == b"new-bytes"
    assert saved.parent == tmp_path.resolve()
