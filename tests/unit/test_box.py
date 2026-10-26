#!/usr/bin/env python3

"""
Unit tests for npointstats.box.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- third party -----
import numpy as np
import pytest

# ----- local modules -----
from npointstats.box import Box

# ============================================================
# CONSTRUCTION AND VALIDATION
# ============================================================


def test_box_stores_read_only_copy() -> None:
    lattice = np.eye(2)
    box = Box(lattice)
    lattice[0, 0] = 5.0
    assert box.lattice[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="read-only"):
        box.lattice[0, 0] = 2.0


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 3)),
        np.zeros((2, 2)),
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.ones((0, 0)),
    ],
)
def test_box_rejects_invalid_lattice(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="lattice"):
        Box(bad)


def test_box_constructors() -> None:
    cube = Box.cubic(2.0, 3)
    np.testing.assert_allclose(cube.lattice, 2.0 * np.eye(3))
    rect = Box.from_lengths([1.0, 3.0])
    np.testing.assert_allclose(rect.lattice, np.diag([1.0, 3.0]))
    assert cube.dimension == 3
    assert rect.is_orthorhombic


def test_box_equality() -> None:
    assert Box.cubic(1.0, 2) == Box(np.eye(2))
    assert Box.cubic(1.0, 2) != Box.cubic(1.0, 3)
    assert Box.cubic(1.0, 2) != "box"


# ============================================================
# GEOMETRY
# ============================================================


def test_volume_and_reciprocal_triangular() -> None:
    box = Box(np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]]))
    assert box.volume == pytest.approx(np.sqrt(3) / 2)
    np.testing.assert_allclose(
        box.lattice @ box.reciprocal.T, 2 * np.pi * np.eye(2), atol=1e-12
    )
    assert not box.is_orthorhombic


def test_inscribed_radius() -> None:
    assert Box.from_lengths([1.0, 4.0]).inscribed_radius == pytest.approx(0.5)
    tri = Box(np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]]))
    # Distance between opposite edges of a unit rhombus is sqrt(3)/2.
    assert tri.inscribed_radius == pytest.approx(np.sqrt(3) / 4)


def test_wrap_into_cell() -> None:
    box = Box.from_lengths([2.0, 3.0])
    x = np.array([[-0.5, 3.5], [2.0, 0.0]])
    np.testing.assert_allclose(box.wrap(x), [[1.5, 0.5], [0.0, 0.0]])


def test_minimum_image_triclinic_translate_is_zero() -> None:
    box = Box(np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]]))
    dx = box.minimum_image(box.lattice[1] - np.zeros(2))
    np.testing.assert_allclose(dx, 0.0, atol=1e-12)


def test_shape_errors() -> None:
    box = Box.cubic(1.0, 2)
    with pytest.raises(ValueError, match="dimension"):
        box.wrap(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="dimension"):
        box.to_fractional(np.zeros(3))
