"""A blank shape picture raises the empty-image error, not a numpy message."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SHAPE = Path(__file__).resolve().parents[1] / "hunyuan_mlx" / "shape"
sys.path.insert(0, str(SHAPE))

from hy3dmlx.preprocess import recenter  # noqa: E402


def test_a_blank_image_says_it_is_empty():
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="input image is empty"):
        recenter(image)
