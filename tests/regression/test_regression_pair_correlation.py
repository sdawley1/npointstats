#!/usr/bin/env python3

"""
Regression tests for pair-correlation defects (D-015, D-017 .. D-020).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import tracemalloc

# ----- third party -----
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

# ----- local modules -----
from npointstats.box import Box
from npointstats.pairs import ball_volume, pair_correlation
from npointstats.pattern import PointPattern

# ============================================================
# CONSTANTS
# ============================================================

TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])

# ============================================================
# D-015: g2 in non-orthogonal boxes
# ============================================================


def test_regression_D015_g2_triclinic_translate() -> None:
    """D-015: two points one lattice vector apart (in a 3x cell) are 1 apart."""
    box = Box(3 * TRIANGULAR)
    pattern = PointPattern([[0.0, 0.0], [0.5, np.sqrt(3) / 2]], box)
    g2 = pair_correlation(pattern, dr=0.07, r_max=1.2)
    assert g2.counts.sum() == 1
    assert g2.counts[14] == 1  # bin [0.98, 1.05)


# ============================================================
# D-017: dr is an absolute width
# ============================================================


@given(st.floats(1.0, 20.0), st.floats(0.01, 0.2))
@example(10.0, 0.1)
def test_regression_D017_dr_is_absolute(length: float, dr: float) -> None:
    """D-017: bins have width dr in position units, whatever the box size."""
    pattern = PointPattern(
        np.random.default_rng(0).random((20, 2)) * length, Box.cubic(length, 2)
    )
    g2 = pair_correlation(pattern, dr=dr, r_max=min(0.5 * length, 50 * dr))
    np.testing.assert_allclose(np.diff(g2.edges), dr, rtol=1e-9)


# ============================================================
# D-018: range must respect r_max and the minimum-image limit
# ============================================================


def test_regression_D018_r_max_respected() -> None:
    """D-018: r_max is honoured and cannot exceed half the shortest box width."""
    x = np.random.default_rng(1).random((50, 2)) * np.array([1.0, 4.0])
    pattern = PointPattern(x, np.diag([1.0, 4.0]))
    g2 = pair_correlation(pattern, dr=0.01, r_max=0.1)
    assert g2.edges[-1] == pytest.approx(0.1)
    # edge: the old default went to max(lv)/2 = 2.0, beyond the valid 0.5
    assert pair_correlation(pattern, dr=0.01).edges[-1] <= 0.5 + 1e-12
    with pytest.raises(ValueError, match="inscribed"):
        pair_correlation(pattern, dr=0.01, r_max=2.0)


# ============================================================
# D-019: exact shell volumes, bin centres
# ============================================================


@pytest.mark.parametrize("d", [1, 2, 3])
def test_regression_D019_exact_shell_normalisation(d: int) -> None:
    """D-019: one pair in bin [0.05, 0.10) gives g2 = V / (N(N-1) dV) exactly."""
    x = np.zeros((2, d))
    x[1, 0] = 0.07
    pattern = PointPattern(x, np.eye(d))
    g2 = pair_correlation(pattern, dr=0.05, r_max=0.1)
    shell = ball_volume(0.10, d) - ball_volume(0.05, d)
    np.testing.assert_allclose(g2.mean[1], 2 / (2 * 1 * shell), rtol=1e-12)
    np.testing.assert_allclose(g2.x_mean, [0.025, 0.075])
    assert g2.mean[0] == 0.0


@given(st.integers(1, 3), st.floats(0.05, 0.4))
@example(2, 0.075)
def test_regression_D019_single_pair_property(d: int, r: float) -> None:
    """D-019 property: a single pair is normalised by its own bin's exact volume."""
    x = np.zeros((2, d))
    x[1, -1] = r
    g2 = pair_correlation(PointPattern(x, np.eye(d)), dr=0.05, r_max=0.45)
    i = int(np.searchsorted(g2.edges, r, side="right") - 1)
    shell = ball_volume(g2.edges[i + 1], d) - ball_volume(g2.edges[i], d)
    np.testing.assert_allclose(g2.mean[i], 1.0 / shell, rtol=1e-12)
    assert g2.counts.sum() == 1


# ============================================================
# D-020: memory must not scale as N^2
# ============================================================


def test_regression_D020_memory_bounded() -> None:
    """D-020: N = 50,000 uses far less than the 20 GB of a dense distance matrix."""
    pattern = PointPattern(np.random.default_rng(2).random((50_000, 2)), np.eye(2))
    tracemalloc.start()
    g2 = pair_correlation(pattern)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert g2.counts.sum() > 0
    assert peak < 4e8
