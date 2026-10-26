#!/usr/bin/env python3

"""
Regression tests for geometry defects (minimum image, argument mutation).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- third party -----
import numpy as np
from hypothesis import example, given
from hypothesis import strategies as st
from npst_strategies import boxes

# ----- local modules -----
from npointstats.box import Box

# ============================================================
# CONSTANTS
# ============================================================

TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])

# ============================================================
# D-015: minimum image in non-orthogonal boxes
# ============================================================


def test_regression_D015_triclinic_minimum_image() -> None:
    """D-015: a lattice translate must have zero minimum-image distance."""
    box = Box(TRIANGULAR)
    dx = box.minimum_image(np.array([0.5, np.sqrt(3) / 2]) - np.zeros(2))
    assert np.linalg.norm(dx) < 1e-12
    # edge: sum of both lattice vectors is also a lattice translate
    dx2 = box.minimum_image(TRIANGULAR.sum(axis=0))
    assert np.linalg.norm(dx2) < 1e-12


@given(boxes(), st.integers(-3, 3), st.integers(-3, 3), st.integers(-3, 3))
@example(Box(TRIANGULAR), 0, 1, 0)
def test_regression_D015_lattice_translates_vanish(
    box: Box, i: int, j: int, k: int
) -> None:
    """D-015 property: every lattice vector maps to the zero displacement."""
    n = np.array([i, j, k][: box.dimension], dtype=np.float64)
    dx = box.minimum_image(n @ box.lattice)
    assert np.linalg.norm(dx) < 1e-9 * max(1.0, np.abs(n).sum())


# ============================================================
# D-016: wrapping must not mutate its argument
# ============================================================


def test_regression_D016_wrap_does_not_mutate() -> None:
    """D-016: wrapping a point returns a new array and leaves the input alone."""
    p = np.array([1.5, 0.2])
    out = Box.cubic(1.0, 2).wrap(p)
    np.testing.assert_allclose(p, [1.5, 0.2])
    np.testing.assert_allclose(out, [0.5, 0.2])
    # edge: a point exactly on the boundary
    q = np.array([1.0, -0.0])
    Box.cubic(1.0, 2).wrap(q)
    np.testing.assert_allclose(q, [1.0, 0.0])


@given(boxes(), st.data())
@example(Box.cubic(1.0, 2), None)
def test_regression_D016_no_mutation_property(
    box: Box, data: st.DataObject | None
) -> None:
    """D-016 property: wrap/minimum_image/to_fractional never edit their input."""
    if data is None:
        x = np.array([[1.5, 0.2]])
    else:
        coords = st.floats(-5.0, 5.0, allow_nan=False, allow_subnormal=False)
        x = np.array(
            [[data.draw(coords) for _ in range(box.dimension)] for _ in range(3)]
        )
    before = x.copy()
    box.wrap(x)
    box.minimum_image(x)
    box.to_fractional(x)
    np.testing.assert_array_equal(x, before)
