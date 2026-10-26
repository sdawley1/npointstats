#!/usr/bin/env python3

"""
Unit and property tests for npointstats.pair_correlation.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import itertools

# ----- third party -----
import numpy as np
import pytest
from hypothesis import given
from npst_strategies import brute_force_min_image, pattern_and_shift, patterns

# ----- local modules -----
from npointstats.box import Box
from npointstats.pair_correlation import (
    ball_volume,
    default_r_max,
    pair_correlation,
    pair_counts,
)
from npointstats.pattern import PointPattern

# ============================================================
# HELPERS
# ============================================================


def brute_force_distances(pattern: PointPattern) -> np.ndarray:
    """All N(N-1) ordered-pair distances from brute-force minimum images."""
    x = pattern.positions
    dx = (x[:, None, :] - x[None, :, :]).reshape(-1, pattern.dimension)
    r = np.linalg.norm(brute_force_min_image(pattern.box, dx), axis=1)
    r = r.reshape(pattern.n_points, pattern.n_points)
    return np.asarray(r[~np.eye(pattern.n_points, dtype=bool)])


def brute_force_counts(pattern: PointPattern, edges: np.ndarray) -> np.ndarray:
    """Ordered-pair histogram from all N^2 brute-force minimum images."""
    r = brute_force_distances(pattern)
    idx = np.searchsorted(edges, r, side="right") - 1
    nb = edges.size - 1
    return np.bincount(idx[(idx >= 0) & (idx < nb)], minlength=nb)


# ============================================================
# UNIT TESTS
# ============================================================


def test_ball_volume() -> None:
    np.testing.assert_allclose(ball_volume(2.0, 1), 4.0)
    np.testing.assert_allclose(ball_volume(1.0, 2), np.pi)
    np.testing.assert_allclose(ball_volume(1.0, 3), 4 * np.pi / 3)
    with pytest.raises(ValueError, match="dimension"):
        ball_volume(1.0, 0)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_poisson_g2_is_one(d: int, rng: np.random.Generator) -> None:
    n = {1: 4000, 2: 4000, 3: 4000}[d]
    pattern = PointPattern(rng.random((n, d)), np.eye(d))
    g2 = pair_correlation(pattern, dr=0.02 if d < 3 else 0.05, r_max=0.4)
    # pooled estimate over all bins: binomial pattern has E[g2] = 1 exactly
    pooled = np.sum(g2.counts) / np.sum(g2.counts / g2.mean)
    assert abs(pooled - 1.0) < 0.02
    z = (g2.mean - 1.0) / g2.sem
    assert np.mean(np.abs(z) < 3) > 0.95


def test_square_lattice_shells() -> None:
    m = 10
    grid = np.array(list(itertools.product(range(m), repeat=2)), dtype=float)
    pattern = PointPattern(grid, Box.cubic(float(m), 2))
    # shell radii must not sit on bin edges (a one-ulp change would move them)
    g2 = pair_correlation(pattern, dr=0.03, r_max=2.2)
    shells = {1.0: 4, np.sqrt(2): 4, 2.0: 4}
    for r, z in shells.items():
        i = int(np.searchsorted(g2.edges, r, side="right") - 1)
        assert g2.counts[i] == m * m * z // 2
    assert g2.counts.sum() == m * m * 12 // 2


def test_triangular_lattice_in_rhombic_cell() -> None:
    m = 8
    box = Box(m * np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]]))
    s = np.array(list(itertools.product(range(m), repeat=2)), dtype=float) / m
    pattern = PointPattern(box.to_cartesian(s), box)
    g2 = pair_correlation(pattern, dr=0.07, r_max=1.5)
    # six neighbours at r = 1 (bin [0.98, 1.05)), none elsewhere below sqrt 3
    assert g2.counts.sum() == m * m * 6 // 2
    assert g2.counts[14] == m * m * 6 // 2


def test_defaults_and_errors(rng: np.random.Generator) -> None:
    pattern = PointPattern(rng.random((100, 2)), np.eye(2))
    assert default_r_max(pattern) == pytest.approx(0.5)
    big = PointPattern(rng.random((10_000, 2)), np.eye(2))
    assert default_r_max(big) == pytest.approx(0.1)
    g2 = pair_correlation(pattern)
    assert g2.metadata["statistic"] == "pair_correlation"
    with pytest.raises(ValueError, match="two points"):
        pair_correlation(PointPattern([[0.1, 0.1]], np.eye(2)))
    with pytest.raises(ValueError, match="inscribed"):
        pair_counts(pattern, np.array([0.0, 0.6]))
    with pytest.raises(ValueError, match="increasing"):
        pair_counts(pattern, np.array([0.0, 0.0]))


def test_nonzero_first_edge_excludes_self_pairs() -> None:
    pattern = PointPattern([[0.0, 0.0], [0.0, 0.0], [0.3, 0.0]], np.eye(2))
    np.testing.assert_array_equal(pair_counts(pattern, np.array([0.0, 0.1])), [2])
    np.testing.assert_array_equal(pair_counts(pattern, np.array([0.1, 0.4])), [4])


# ============================================================
# PROPERTY TESTS
# ============================================================


@given(patterns(min_n=2, max_n=25))
def test_pair_counts_match_brute_force(pattern: PointPattern) -> None:
    r_in = pattern.box.inscribed_radius
    edges = np.linspace(0.0, r_in, 7)
    got = pair_counts(pattern, edges)
    want = brute_force_counts(pattern, edges)
    # distances within rounding of an edge may land on either side (e.g. a pair
    # exactly L/2 apart, where two images tie); nothing else may differ
    r = brute_force_distances(pattern)
    near = np.abs(r[:, None] - edges[None, :]) <= 1e-12 * max(1.0, r_in)
    assert np.abs(got - want).sum() <= 2 * near.any(axis=1).sum()
    if not near.any():
        np.testing.assert_array_equal(got, want)


@given(patterns(min_n=2, max_n=25))
def test_g2_bounds_and_symmetry(pattern: PointPattern) -> None:
    g2 = pair_correlation(pattern, dr=pattern.box.inscribed_radius / 5)
    assert np.all(g2.mean >= 0)
    # ordered counts are even: every pair (i, j) has its partner (j, i)
    total = pair_counts(pattern, g2.edges)
    assert np.all(total % 2 == 0)


@given(pattern_and_shift(max_n=25))
def test_g2_translation_invariance(case: tuple[PointPattern, np.ndarray]) -> None:
    pattern, shift = case
    if pattern.n_points < 2:
        return
    edges = np.linspace(0.0, pattern.box.inscribed_radius, 6)
    moved = PointPattern(pattern.positions + shift, pattern.box)
    # translation can move a distance across a bin edge by one ulp; compare
    # cumulative counts away from the edges
    a = pair_counts(pattern, edges)
    b = pair_counts(moved, edges)
    assert abs(int(a.sum()) - int(b.sum())) <= 2 * pattern.n_points
