#!/usr/bin/env python3

"""
Integration tests: independent estimators of the same quantity must agree.

- window-sampled sigma^2(R) vs the S(k) torus identity, configuration by
  configuration;
- S(k) of a perturbed lattice vs its analytic small-displacement limit.

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

# ----- local modules -----
from npointstats.box import Box
from npointstats.number_variance import number_variance, number_variance_from_sk
from npointstats.pattern import PointPattern
from npointstats.structure_factor import structure_factor

# ============================================================
# HELPERS
# ============================================================


def perturbed_lattice(
    m: int, d: int, sigma: float, rng: np.random.Generator
) -> PointPattern:
    """m^d points on a unit-cell grid with Gaussian displacements of width sigma/m."""
    s = np.array(list(itertools.product(range(m), repeat=d)), dtype=float) / m
    s += (sigma / m) * rng.standard_normal(s.shape)
    return PointPattern(s % 1.0, Box.cubic(1.0, d))


# ============================================================
# TESTS
# ============================================================


@pytest.mark.parametrize(
    ("d", "kind"),
    [(1, "poisson"), (2, "poisson"), (2, "lattice"), (3, "poisson"), (3, "lattice")],
)
def test_window_variance_matches_structure_factor_identity(
    d: int, kind: str, rng: np.random.Generator
) -> None:
    n_side = {1: 400, 2: 30, 3: 9}[d]
    if kind == "poisson":
        pattern = PointPattern(rng.random((n_side**d, d)), np.eye(d))
    else:
        pattern = perturbed_lattice(n_side, d, 0.1, rng)
    radii = np.array([0.6, 1.2, 2.5]) / n_side
    windows = number_variance(pattern, radii, n_windows=30000, seed=11)
    sk = structure_factor(pattern, k_max=2 * np.pi * n_side * {1: 40, 2: 8, 3: 4}[d])
    exact = number_variance_from_sk(sk, radii)
    z = (windows.variance - exact.variance) / windows.sem
    assert np.all(np.abs(z) < 5), z
    np.testing.assert_allclose(windows.mean_count, exact.mean_count, rtol=0.05)


def test_ideal_tail_improves_truncated_sum(rng: np.random.Generator) -> None:
    pattern = PointPattern(rng.random((500, 2)), np.eye(2))
    radii = np.array([0.05])
    sk = structure_factor(pattern, k_max=2 * np.pi * 15)
    with_tail = number_variance_from_sk(sk, radii, ideal_tail=True).variance
    truncated = number_variance_from_sk(sk, radii, ideal_tail=False).variance
    windows = number_variance(pattern, radii, n_windows=40000, seed=2)
    assert abs(with_tail - windows.variance) < abs(truncated - windows.variance)


def test_perturbed_lattice_small_k_limit(rng: np.random.Generator) -> None:
    """Gaussian-displaced lattice: E[S(k)] = 1 - exp(-k^2 s^2) below the Bragg peaks."""
    m, sigma, members = 24, 0.05, 40
    total = None
    for _ in range(members):
        sk = structure_factor(perturbed_lattice(m, 2, sigma, rng), k_max=2 * np.pi * 6)
        total = sk.values if total is None else total + sk.values
    assert total is not None
    ratio = (total / members) / (1.0 - np.exp(-(sk.k**2) * (sigma / m) ** 2))
    # each shell holds 4-16 independent exponential variates per member
    for shell in range(1, 6):
        sel = np.abs(sk.k - 2 * np.pi * shell) < np.pi
        assert abs(ratio[sel].mean() - 1.0) < 0.12
