#!/usr/bin/env python3

"""
Regression tests for structure-factor defects.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import itertools
import tracemalloc

# ----- third party -----
import numpy as np
from hypothesis import example, given
from hypothesis import strategies as st
from npst_strategies import boxes, patterns

# ----- local modules -----
from npointstats.box import Box
from npointstats.pattern import PointPattern
from npointstats.structure_factor import structure_factor, wavevector_indices

# ============================================================
# CONSTANTS
# ============================================================

TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])

# ============================================================
# D-001: caller's coordinates must not be rescaled
# ============================================================


def test_regression_D001_positions_not_mutated() -> None:
    """D-001: computing S(k) leaves the caller's array untouched (non-unit box)."""
    x = 2.0 * np.random.default_rng(1).random((10, 3))
    before = x.copy()
    structure_factor(PointPattern(x, 2.0 * np.eye(3)), k_max=10.0)
    np.testing.assert_array_equal(x, before)
    # edge: the direct method and an anisotropic box
    y = x * np.array([1.0, 3.0, 0.5])
    y_before = y.copy()
    box = np.diag([2.0, 6.0, 1.0])
    structure_factor(PointPattern(y, box), k_max=8.0, method="direct")
    np.testing.assert_array_equal(y, y_before)


@given(patterns())
@example(PointPattern(np.full((10, 3), 1.5), 2.0 * np.eye(3)))
def test_regression_D001_no_mutation_property(pattern: PointPattern) -> None:
    """D-001 property: the pattern's positions are identical after the call."""
    before = pattern.positions.copy()
    structure_factor(pattern, k_max=4 * np.pi / pattern.box.inscribed_radius)
    np.testing.assert_array_equal(pattern.positions, before)


# ============================================================
# D-004: the requested k range must be honoured
# ============================================================


def test_regression_D004_k_max_respected() -> None:
    """D-004: a small k_max gives the small set of vectors, not a fixed 25^3 grid."""
    x = np.random.default_rng(2).random((10, 3))
    sk = structure_factor(PointPattern(x, np.eye(3)), k_max=2 * np.pi * 2.0)
    # integer points of a radius-2 ball minus the origin: 33 - 1 = 32
    assert sk.values.size == 32
    assert sk.k.max() <= 2 * np.pi * 2.0 + 1e-9
    # edge: k_max just below the first shell -> error, not a silent default grid
    one = structure_factor(PointPattern(x, np.eye(3)), k_max=2 * np.pi * 1.0)
    assert one.values.size == 6


@given(boxes(), st.floats(1.0, 4.0))
@example(Box.cubic(1.0, 3), 2.0)
def test_regression_D004_count_matches_enumeration(box: Box, n_shells: float) -> None:
    """D-004 property: returned vectors are exactly the lattice points in the ball."""
    k_max = n_shells * float(np.max(np.linalg.norm(box.reciprocal, axis=1)))
    got = wavevector_indices(box, k_max)
    # brute force over a generous index cube: |n_i| = |k . a_i| / 2 pi
    # <= k_max |a_i| / 2 pi by Cauchy-Schwarz; add a margin of 2
    reach = int(k_max * np.max(np.linalg.norm(box.lattice, axis=1)) / (2 * np.pi)) + 2
    grid = np.array(
        list(itertools.product(range(-reach, reach + 1), repeat=box.dimension)),
        dtype=float,
    )
    k = np.linalg.norm(grid @ box.reciprocal, axis=1)
    want = int(np.sum((k > 0) & (k <= k_max * (1 + 1e-12))))
    assert got.shape[0] == want


# ============================================================
# D-011: wave vectors with negative components must be sampled
# ============================================================


def test_regression_D011_negative_components_present() -> None:
    """D-011: all quadrants of k-space are sampled, so S(kx, ky) != S(-kx, ky) shows."""
    box = Box.cubic(1.0, 2)
    n = wavevector_indices(box, 2 * np.pi * 3)
    assert np.any((n[:, 0] < 0) & (n[:, 1] > 0))
    assert np.any((n[:, 0] < 0) & (n[:, 1] < 0))
    # edge: an anisotropic pattern whose S differs between (1, 1) and (-1, 1)
    pattern = PointPattern([[0.0, 0.0], [0.25, 0.25]], box)
    sk = structure_factor(pattern, k_max=2 * np.pi * 1.5)
    idx = np.rint(sk.k_vectors / (2 * np.pi)).astype(int).tolist()
    s_pp = sk.values[idx.index([1, 1])]
    s_mp = sk.values[idx.index([-1, 1])]
    np.testing.assert_allclose([s_pp, s_mp], [0.0, 2.0], atol=1e-9)


# ============================================================
# D-012: radial bins must hold complete shells
# ============================================================


@given(boxes(dimension=2), st.integers(2, 6))
@example(Box.cubic(1.0, 2), 5)
def test_regression_D012_complete_shells(box: Box, n_bins: int) -> None:
    """D-012 property: each radial bin holds every lattice vector of its shell."""
    x = np.zeros((1, 2))
    dk = float(np.min(np.linalg.norm(box.reciprocal, axis=1)))
    sk = structure_factor(PointPattern(x, box), k_max=(n_bins + 0.5) * dk)
    res = sk.radial_average(dk=dk)
    reach = (
        int(sk.k_max * np.max(np.linalg.norm(box.lattice, axis=1)) / (2 * np.pi)) + 2
    )
    grid = np.array(list(itertools.product(range(-reach, reach + 1), repeat=2)), float)
    k = np.linalg.norm(grid @ box.reciprocal, axis=1)
    for lo, hi, count in zip(res.edges[:-1], res.edges[1:], res.counts, strict=True):
        assert count == int(np.sum((k >= lo) & (k < hi) & (k > 0)))


# ============================================================
# D-013: bounded memory
# ============================================================


def test_regression_D013_direct_sum_memory_bounded() -> None:
    """D-013: the direct sum never materialises the full (M, N) phase matrix."""
    rng = np.random.default_rng(3)
    pattern = PointPattern(rng.random((10_000, 2)), np.eye(2))
    k_max = 2 * np.pi * 30.0
    m = wavevector_indices(pattern.box, k_max).shape[0]
    full_bytes = m * pattern.n_points * 24  # float phase + complex exponential
    tracemalloc.start()
    structure_factor(pattern, k_max=k_max, method="direct")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert full_bytes > 5e8
    assert peak < 0.4 * full_bytes
    assert peak < 2.5e8


# ============================================================
# D-014: non-orthogonal boxes
# ============================================================


def test_regression_D014_triclinic_bragg_peaks() -> None:
    """D-014: a triangular crystal in its rhombic cell peaks only at n = 0 mod m."""
    m = 6
    box = Box(m * TRIANGULAR)
    s = np.array(list(itertools.product(range(m), repeat=2)), dtype=float) / m
    pattern = PointPattern(box.to_cartesian(s), box)
    first_shell = 4 * np.pi / np.sqrt(3)
    sk = structure_factor(pattern, k_max=1.2 * first_shell)
    n = np.rint(sk.k_vectors @ box.lattice.T / (2 * np.pi)).astype(int)
    bragg = np.all(n % m == 0, axis=1)
    # first Bragg shell of a unit triangular lattice: |k| = 4 pi / sqrt(3), 6 peaks
    assert bragg.sum() == 6
    np.testing.assert_allclose(sk.k[bragg], first_shell)
    np.testing.assert_allclose(sk.values[bragg], m * m, rtol=1e-9)
    np.testing.assert_allclose(sk.values[~bragg], 0.0, atol=1e-8)
