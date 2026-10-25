"""Paint refuses a checkpoint that is missing a tensor or has one spare."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PAINT = Path(__file__).resolve().parents[1] / "hunyuan_mlx" / "paint"
sys.path.insert(0, str(PAINT))

from hy3dpaint_mlx.convert import require_exact_weight_keys, renamed_weight_keys  # noqa: E402


def test_dropping_one_tensor_is_an_error():
    module_keys = ["down.weight", "up.weight"]
    state_keys = renamed_weight_keys({"down.weight": object()}, ())
    with pytest.raises(ValueError, match="missing"):
        require_exact_weight_keys(module_keys, state_keys)


def test_an_extra_tensor_is_an_error():
    module_keys = ["down.weight"]
    state_keys = renamed_weight_keys(
        {"down.weight": object(), "up.weight": object()}, ()
    )
    with pytest.raises(ValueError, match="extra"):
        require_exact_weight_keys(module_keys, state_keys)


def test_a_rename_is_compared_after_the_rename():
    state_keys = renamed_weight_keys(
        {"attn.to_out.0.weight": object()}, ((".to_out.0.", ".to_out."),)
    )
    require_exact_weight_keys(["attn.to_out.weight"], state_keys)
