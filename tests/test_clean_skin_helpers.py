"""The skin cleaner removes only the known helper."""

from scripts.clean_skin_weights import is_export_helper


def test_a_prefixed_name_is_kept():
    assert is_export_helper("Icosphere.001") is False
    assert is_export_helper("Icosphere_body") is False


def test_the_exact_helper_name_is_removed():
    assert is_export_helper("Icosphere") is True


def test_the_gltf_extra_is_removed_under_any_name():
    assert is_export_helper("Body", {"glTF_not_exported": True}) is True
