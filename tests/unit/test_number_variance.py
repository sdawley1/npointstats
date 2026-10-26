#!/usr/bin/env python3

"""
Unit and property tests for npointstats.number_variance.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- third party -----
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from npst_strategies import brute_force_min_image, patterns, unit_float
from scipy.integrate import quad

# ----- local modules -----
from npointstats.box import Box
from npointstats.number_variance import (
    NumberVariance,
    alpha_tilde,
    default_radii,
    number_variance,
    window_counts,
)
from npointstats.pair_correlation import ball_volume
from npointstats.pattern import PointPattern

# ============================================================
# HELPERS
# ============================================================


def brute_force_window_counts(
    pattern: PointPattern, centers: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """Counts from all window-point minimum images (open balls)."""
    w, n = centers.shape[0], pattern.n_points
    dx = (pattern.positions[None, :, :] - centers[:, None, :]).reshape(
        -1, pattern.dimension
    )
    r = np.linalg.norm(brute_force_min_image(pattern.box, dx), axis=1).reshape(w, n)
    return np.stack([(r < R).sum(axis=1) for R in radii], axis=1)


# ============================================================
# UNIT TESTS
# ============================================================


def test_alpha_tilde_normalisation() -> None:
    """alpha(0) = v(R) and (2 pi)^-d int alpha d^dk = 1 (alpha_2(r=0) = 1)."""
    for d, area in ((1, 2.0), (2, 2 * np.pi), (3, 4 * np.pi)):
        radius = 0.7
        assert alpha_tilde(np.array([0.0]), radius, d)[0] == pytest.approx(
            float(ball_volume(radius, d))
        )

        def integrand(
            k: float, d: int = d, area: float = area, rad: float = radius
        ) -> float:
            return float(area * k ** (d - 1) * alpha_tilde(np.array([k]), rad, d)[0])

        total = quad(integrand, 0, 400, limit=4000)[0] / (2 * np.pi) ** d
        assert total == pytest.approx(1.0, abs=5e-3)


def test_one_dimensional_lattice_exact_variance() -> None:
    """Unit-spaced 1D lattice: windows of half-width R hold floor or ceil(2R)."""
    pattern = PointPattern(np.arange(20.0)[:, None], Box.cubic(20.0, 1))
    nv = number_variance(pattern, radii=[0.25, 0.5, 0.75], n_windows=20000, seed=3)
    # window of length 2R in a unit lattice: count is 0/1 (R=0.25: p = 0.5),
    # exactly 1 (R = 0.5), or 1/2 (R = 0.75: p = 0.5)
    np.testing.assert_allclose(nv.variance, [0.25, 0.0, 0.25], atol=0.01)
    np.testing.assert_allclose(nv.mean_count, [0.5, 1.0, 1.5], atol=0.02)


def test_poisson_is_binomial(rng: np.random.Generator) -> None:
    """Ensemble-averaged window variance of a binomial pattern is N p (1 - p)."""
    radii = np.array([0.02, 0.05, 0.1])
    n, members = 1000, 30
    variances = [
        number_variance(
            PointPattern(rng.random((n, 2)), np.eye(2)), radii, n_windows=4000, seed=i
        ).variance
        for i in range(members)
    ]
    p = ball_volume(radii, 2)
    expected = n * p * (1 - p)
    # a single configuration has only ~1/(pi R^2) independent windows, so its
    # variance scatters by ~10%; the 30-member mean scatters by ~2%
    assert np.all(np.abs(np.mean(variances, axis=0) - expected) / expected < 0.08)


def test_seed_reproducible_and_defaults(rng: np.random.Generator) -> None:
    pattern = PointPattern(rng.random((200, 2)), np.eye(2))
    a = number_variance(pattern, n_windows=500, seed=7)
    b = number_variance(pattern, n_windows=500, seed=7)
    np.testing.assert_array_equal(a.variance, b.variance)
    assert a.radii.size == 40
    assert a.radii[-1] <= pattern.box.inscribed_radius
    np.testing.assert_allclose(default_radii(pattern)[-1], 5 * 200**-0.5)
    assert a.to_array().shape == (40, 4)


def test_errors(rng: np.random.Generator) -> None:
    pattern = PointPattern(rng.random((20, 2)), np.eye(2))
    with pytest.raises(ValueError, match="n_windows"):
        number_variance(pattern, n_windows=1)
    with pytest.raises(ValueError, match="positive"):
        number_variance(pattern, radii=[0.0, 0.1])
    with pytest.raises(ValueError, match="increasing"):
        number_variance(pattern, radii=[0.2, 0.1])
    with pytest.raises(ValueError, match="inscribed"):
        number_variance(pattern, radii=[0.6])
    with pytest.raises(ValueError, match="shape"):
        NumberVariance(np.zeros(2), np.zeros(1), np.zeros(2), np.zeros(2))


# ============================================================
# PROPERTY TESTS
# ============================================================


@given(patterns(max_n=25), st.data())
def test_window_counts_match_brute_force(
    pattern: PointPattern, data: st.DataObject
) -> None:
    w = data.draw(st.integers(1, 8))
    s = data.draw(hnp.arrays(np.float64, (w, pattern.dimension), elements=unit_float))
    centers = pattern.box.to_cartesian(s)
    radii = np.linspace(0.1, 0.95, 5) * pattern.box.inscribed_radius
    got = window_counts(pattern, centers, radii)
    want = brute_force_window_counts(pattern, centers, radii)
    # a point at distance exactly R is excluded by both; allow one-ulp edge cases
    assert np.abs(got - want).max() <= 1
    assert np.mean(got == want) > 0.97
    assert np.all(np.diff(got, axis=1) >= 0)


@given(patterns(min_n=2, max_n=25), st.integers(0, 2**31 - 1))
def test_variance_nonnegative_and_mean_bounded(
    pattern: PointPattern, seed: int
) -> None:
    radii = np.linspace(0.2, 1.0, 4) * pattern.box.inscribed_radius
    nv = number_variance(pattern, radii, n_windows=50, seed=seed)
    assert np.all(nv.variance >= 0)
    assert np.all(nv.mean_count <= pattern.n_points)
    assert np.all(np.diff(nv.mean_count) >= 0)
