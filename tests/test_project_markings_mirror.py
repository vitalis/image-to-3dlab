"""Mirror across the mesh, not the world origin."""

import numpy as np

from scripts.project_markings import mirrored_vertices


def test_an_off_centre_mesh_mirrors_through_its_own_middle():
    vertices = np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    mirrored = mirrored_vertices(vertices, "z")
    assert mirrored[:, 0] == pytest_approx([4.0, 2.0])
    assert mirrored[:, 1] == pytest_approx([0.0, 0.0])


def test_looking_along_x_does_not_flip_depth():
    vertices = np.array([[5.0, 0.0, 1.0], [5.0, 0.0, 3.0]])
    mirrored = mirrored_vertices(vertices, "x")
    assert mirrored[:, 0] == pytest_approx([5.0, 5.0])
    assert mirrored[:, 2] == pytest_approx([3.0, 1.0])


def pytest_approx(values):
    import pytest

    return pytest.approx(values)
