#!/usr/bin/env python3

"""
Unit tests for npointstats.pattern.

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
from npointstats.pattern import PointPattern

# ============================================================
# TESTS
# ============================================================


def test_pattern_basic_properties() -> None:
    pattern = PointPattern(np.zeros((4, 3)), 2.0 * np.eye(3))
    assert pattern.n_points == 4
    assert len(pattern) == 4
    assert pattern.dimension == 3
    assert pattern.density == pytest.approx(0.5)
    assert "n_points=4" in repr(pattern)


def test_pattern_accepts_1d_vector() -> None:
    pattern = PointPattern(np.array([0.1, 0.2, 0.3]), Box.cubic(1.0, 1))
    assert pattern.positions.shape == (3, 1)


def test_pattern_positions_read_only() -> None:
    pattern = PointPattern(np.zeros((2, 2)), np.eye(2))
    with pytest.raises(ValueError, match="read-only"):
        pattern.positions[0, 0] = 1.0


@pytest.mark.parametrize(
    ("positions", "match"),
    [
        (np.zeros((3, 3)), "shape"),
        (np.zeros((0, 2)), "at least one"),
        (np.array([[np.inf, 0.0]]), "finite"),
    ],
)
def test_pattern_rejects_bad_positions(positions: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PointPattern(positions, np.eye(2))


def test_wrapped_and_fractional() -> None:
    pattern = PointPattern(np.array([[-0.25, 2.5]]), Box.from_lengths([1.0, 2.0]))
    np.testing.assert_allclose(pattern.wrapped(), [[0.75, 0.5]])
    np.testing.assert_allclose(pattern.fractional(), [[0.75, 0.25]])
