"""Texture inpaint must not hide an OpenCV failure that is not a missing import."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PAINT = Path(__file__).resolve().parents[1] / "hunyuan_mlx" / "paint"
sys.path.insert(0, str(PAINT))

from hy3dpaint_mlx.mesh_render import smooth_inpaint_holes  # noqa: E402


def test_a_runtime_error_from_inpaint_is_not_swallowed():
    def broken(_image, _mask, _radius, _flag):
        raise RuntimeError("opencv broke")

    with pytest.raises(RuntimeError, match="opencv broke"):
        smooth_inpaint_holes(
            broken, np.zeros((2, 2), np.uint8), np.ones((2, 2), np.uint8)
        )


def test_a_missing_opencv_keeps_the_nearest_fill():
    def missing(_image, _mask, _radius, _flag):
        raise ImportError("no cv2")

    image = np.arange(4, dtype=np.uint8).reshape(2, 2)
    assert smooth_inpaint_holes(missing, image, np.zeros((2, 2), np.uint8)) is None
