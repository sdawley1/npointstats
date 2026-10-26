#!/usr/bin/env python3

"""
Unit tests for npointstats.structure.

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
from npointstats.pattern import PointPattern
from npointstats.structure import (
    default_dk,
    default_k_max,
    shell_edges,
    structure_factor,
    wavevector_indices,
)

# ============================================================
# HELPERS
# ============================================================

TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def sublattice(box: Box, m: int) -> PointPattern:
    """m^d points on the fractional grid (i/m, j/m, ...) of `box`."""
    d = box.dimension
    s = np.array(list(itertools.product(range(m), repeat=d)), dtype=np.float64) / m
    return PointPattern(box.to_cartesian(s), box)


def fcc(m: int, a: float = 1.0) -> PointPattern:
    """FCC crystal of m^3 conventional cells (4 m^3 points) in a cubic box."""
    cells = np.array(list(itertools.product(range(m), repeat=3)), dtype=np.float64)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pts = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    return PointPattern(pts, Box.cubic(m * a, 3))


# ============================================================
# WAVE VECTORS
# ============================================================


@pytest.mark.parametrize("lattice", [np.eye(2), np.diag([1.0, 2.5]), TRIANGULAR])
def test_wavevectors_match_brute_force_enumeration(lattice: np.ndarray) -> None:
    box = Box(lattice)
    k_max = 25.0
    got = {tuple(n) for n in wavevector_indices(box, k_max)}
    want = set()
    for n in itertools.product(range(-20, 21), repeat=2):
        k = np.linalg.norm(np.array(n) @ box.reciprocal)
        if 0 < k <= k_max:
            want.add(n)
    assert got == want
    assert all(tuple(-np.array(n)) in got for n in got)


def test_wavevectors_sorted_and_errors() -> None:
    box = Box.cubic(1.0, 3)
    k = np.linalg.norm(wavevector_indices(box, 20.0) @ box.reciprocal, axis=1)
    assert np.all(np.diff(k) >= 0)
    with pytest.raises(ValueError, match="positive"):
        wavevector_indices(box, 0.0)
    with pytest.raises(ValueError, match="modes"):
        wavevector_indices(box, 1e6)


def test_defaults() -> None:
    pattern = PointPattern(np.random.default_rng(0).random((100, 2)), np.eye(2))
    assert default_k_max(pattern) == pytest.approx(3 * 2 * np.pi * 10)
    assert default_dk(Box.from_lengths([1.0, 2.0])) == pytest.approx(np.pi)
    np.testing.assert_allclose(shell_edges(1.0, 3.6), [0.5, 1.5, 2.5, 3.5])
    with pytest.raises(ValueError, match="smaller"):
        shell_edges(1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        shell_edges(-1.0, 1.0)


# ============================================================
# CRYSTALS
# ============================================================


@pytest.mark.parametrize("method", ["nufft", "direct"])
@pytest.mark.parametrize(
    "lattice", [np.eye(1), np.eye(2), np.diag([1.0, 1.7]), TRIANGULAR, np.eye(3)]
)
def test_bragg_peaks_of_sublattice(lattice: np.ndarray, method: str) -> None:
    box = Box(lattice)
    m = 4
    pattern = sublattice(box, m)
    sk = structure_factor(pattern, k_max=3.0 * 2 * np.pi * m, method=method)
    n = np.rint(sk.k_vectors @ box.lattice.T / (2 * np.pi)).astype(int)
    bragg = np.all(n % m == 0, axis=1)
    assert bragg.any()
    np.testing.assert_allclose(sk.values[bragg], pattern.n_points, rtol=1e-9)
    np.testing.assert_allclose(sk.values[~bragg], 0.0, atol=1e-9 * pattern.n_points)


def test_fcc_bragg_peaks() -> None:
    m = 3
    pattern = fcc(m)
    sk = structure_factor(pattern, k_max=2 * np.pi * 2.5)
    h = np.rint(sk.k_vectors / (2 * np.pi / m)).astype(int)
    on = np.all(h % m == 0, axis=1)
    hkl = h // m
    parity = hkl % 2
    allowed = on & (np.all(parity == 0, axis=1) | np.all(parity == 1, axis=1))
    np.testing.assert_allclose(sk.values[allowed], pattern.n_points, rtol=1e-9)
    np.testing.assert_allclose(sk.values[~allowed], 0.0, atol=1e-8)
    # smallest allowed reflection is {111} at |k| = 2 pi sqrt(3)
    assert sk.k[allowed].min() == pytest.approx(2 * np.pi * np.sqrt(3))


# ============================================================
# DISORDER AND BINNING
# ============================================================


@pytest.mark.parametrize("d", [1, 2, 3])
def test_poisson_mean_is_one(d: int, rng: np.random.Generator) -> None:
    pattern = PointPattern(rng.random((3000, d)), np.eye(d))
    sk = structure_factor(pattern, k_max=default_k_max(pattern) / 3)
    m = sk.values.size
    # S(k) of a Poisson pattern is exponentially distributed with mean 1;
    # k and -k are identical, so only m/2 values are independent
    assert abs(sk.values.mean() - 1.0) < 6.0 / np.sqrt(m / 2)


def test_radial_average_complete_shells(rng: np.random.Generator) -> None:
    pattern = PointPattern(rng.random((500, 2)), np.eye(2))
    sk = structure_factor(pattern, k_max=200.0)
    res = sk.radial_average()
    assert res.edges[-1] <= 200.0
    assert res.metadata["statistic"] == "structure_factor"
    assert res.counts.sum() == np.sum(sk.k < res.edges[-1])
    coarse = sk.radial_average(dk=4 * np.pi, k_max=100.0)
    assert coarse.edges[-1] <= 100.0


def test_structure_factor_errors() -> None:
    pattern = PointPattern([[0.1, 0.2]], np.eye(2))
    with pytest.raises(ValueError, match="method"):
        structure_factor(pattern, method="gpu")
    with pytest.raises(ValueError, match="no reciprocal"):
        structure_factor(pattern, k_max=1.0)
    four_d = PointPattern(np.zeros((1, 4)), np.eye(4))
    with pytest.raises(ValueError, match="d = 1, 2, 3"):
        structure_factor(four_d, k_max=7.0)
    assert structure_factor(four_d, k_max=7.0, method="direct").values.size == 8
