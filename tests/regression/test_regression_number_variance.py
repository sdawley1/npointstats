#!/usr/bin/env python3

"""
Regression tests for number-variance defects (D-022).

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

# ----- local modules -----
from npointstats.box import Box
from npointstats.number_variance import number_variance, window_counts
from npointstats.pattern import PointPattern

# ============================================================
# D-022: 2D windows, reproducibility
# ============================================================


def test_regression_D022_two_dimensional_windows() -> None:
    """D-022: 2D window counts are exact (no stray z coordinate) and seeded."""
    pts = np.array([[0.1, 0.1], [0.12, 0.1], [0.9, 0.1], [0.5, 0.5]])
    pattern = PointPattern(pts, Box.cubic(1.0, 2))
    centers = np.array([[0.1, 0.1], [0.0, 0.1], [0.5, 0.5]])
    counts = window_counts(pattern, centers, [0.05, 0.25])
    # (0.0, 0.1) sees 0.1 and 0.12 directly and 0.9 through the periodic face
    np.testing.assert_array_equal(counts, [[2, 3], [0, 3], [1, 1]])
    # edge: identical seeds give identical results, different seeds differ
    a = number_variance(pattern, [0.1, 0.2], n_windows=200, seed=5)
    b = number_variance(pattern, [0.1, 0.2], n_windows=200, seed=5)
    c = number_variance(pattern, [0.1, 0.2], n_windows=200, seed=6)
    np.testing.assert_array_equal(a.variance, b.variance)
    assert not np.array_equal(a.variance, c.variance)


@given(st.floats(0.0, 1.0, exclude_max=True), st.floats(0.0, 1.0, exclude_max=True))
@example(0.0, 0.1)
def test_regression_D022_single_point_window_property(cx: float, cy: float) -> None:
    """D-022 property: a window contains a lone point iff its periodic distance < R."""
    pattern = PointPattern([[0.3, 0.7]], Box.cubic(1.0, 2))
    radius = 0.2
    counts = window_counts(pattern, np.array([[cx, cy]]), [radius])
    d = np.array([cx, cy]) - np.array([0.3, 0.7])
    d -= np.rint(d)
    inside = np.linalg.norm(d) < radius
    # skip the measure-zero shell |d| == R within rounding
    if abs(np.linalg.norm(d) - radius) > 1e-12:
        assert counts[0, 0] == int(inside)
