#!/usr/bin/env python3

"""
Regression tests for defects found in code review of the refactor (D-047+).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import importlib
import sys

# ----- third party -----
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from npst_strategies import boxes

# ----- local modules -----
import npointstats

# ============================================================
# D-052: no submodule shadowed by a public attribute
# ============================================================


def test_regression_D052_no_submodule_shadowing() -> None:
    """D-052: every submodule is reachable as the attribute of the same name."""
    import pkgutil

    for info in pkgutil.iter_modules(npointstats.__path__):
        module = importlib.import_module(f"npointstats.{info.name}")
        assert sys.modules[f"npointstats.{info.name}"] is module
        attr = getattr(npointstats, info.name, module)
        assert attr is module, f"npointstats.{info.name} is shadowed by {attr!r}"
    # edge: the public functions are still exported at top level
    assert callable(npointstats.structure_factor)
    assert callable(npointstats.pair_correlation)
    assert callable(npointstats.number_variance)
    assert np.isfinite(npointstats.Box.cubic(1.0, 2).volume)


# ============================================================
# D-047 / D-048: results must not depend on the choice of basis
# ============================================================

SHEARED = np.array([[30.0, 0.0], [27.0, 3.0]])
SHEARED_REDUCED = np.array([[-3.0, 3.0], [30.0, 0.0]])


def _pattern(lattice: np.ndarray, n: int, seed: int) -> "npointstats.PointPattern":
    box = npointstats.Box(lattice)
    s = np.random.default_rng(seed).random((n, box.dimension))
    return npointstats.PointPattern(box.to_cartesian(s), box)


def test_regression_D047_default_dk_keeps_smallest_k() -> None:
    """D-047: the default shell width is the shortest |k|; nothing falls below bin 0."""
    sk = npointstats.structure_factor(_pattern(SHEARED, 500, 0))
    res = sk.radial_average()
    assert res.metadata["dk"] == pytest.approx(sk.k.min())
    assert res.counts.sum() == np.sum(sk.k < res.edges[-1])
    # edge: an ensemble mixing the sheared cell with a square cell drops nothing
    ens = npointstats.Ensemble(
        [_pattern(SHEARED, 500, 1), _pattern(30 * np.eye(2), 500, 2)]
    )
    binned = ens.structure_factor()
    # the finest member spacing wins: 2 pi / 30 for the square cell
    assert binned.metadata["dk"] == pytest.approx(min(sk.k.min(), 2 * np.pi / 30))


def test_regression_D048_limits_independent_of_basis() -> None:
    """D-048: equivalent bases give the same radius limit and mode grid."""
    a, b = npointstats.Box(SHEARED), npointstats.Box(SHEARED_REDUCED)
    assert a.inscribed_radius == pytest.approx(b.inscribed_radius)
    assert a.inscribed_radius > 2.0  # was 1.5 for the sheared basis
    # both reduced bases span the same lattice (integer unimodular transform)
    t = a.reduced_lattice @ np.linalg.inv(b.reduced_lattice)
    np.testing.assert_allclose(t, np.rint(t), atol=1e-9)
    assert abs(abs(np.linalg.det(t)) - 1.0) < 1e-9


@st.composite
def unimodular(draw: st.DrawFn, d: int) -> np.ndarray:
    """Product of a few integer shears and swaps (determinant +-1)."""
    u = np.eye(d, dtype=np.int64)
    for _ in range(draw(st.integers(0, 4))):
        i, j = draw(st.integers(0, d - 1)), draw(st.integers(0, d - 1))
        e = np.eye(d, dtype=np.int64)
        if i == j:
            e[i, i] = -1
        else:
            e[i, j] = draw(st.integers(-3, 3))
        u = e @ u
    return u


@given(
    st.integers(1, 3).flatmap(lambda d: st.tuples(boxes(dimension=d), unimodular(d))),
    st.integers(0, 2**16),
)
@example((npointstats.Box(SHEARED_REDUCED), np.array([[1, 0], [-1, 1]])), 0)
@settings(max_examples=60)
def test_regression_D048_change_of_basis_invariance(case: tuple, seed: int) -> None:
    """D-048 property: a unimodular change of basis changes no statistic."""
    box, u = case
    other = npointstats.Box(u.astype(float) @ box.lattice)
    s = np.random.default_rng(seed).random((12, box.dimension))
    p1 = npointstats.PointPattern(box.to_cartesian(s), box)
    p2 = npointstats.PointPattern(p1.positions, other)
    assert other.inscribed_radius == pytest.approx(box.inscribed_radius, rel=1e-9)
    k_max = 3.0 * float(np.max(np.linalg.norm(box.reduced_reciprocal, axis=1)))
    a = npointstats.structure_factor(p1, k_max=k_max)
    b = npointstats.structure_factor(p2, k_max=k_max)
    assert a.values.size == b.values.size
    np.testing.assert_allclose(
        np.sort(a.values), np.sort(b.values), rtol=1e-8, atol=1e-8
    )
    edges = np.linspace(0.0, box.inscribed_radius, 5)
    np.testing.assert_array_equal(
        npointstats.pairs.pair_counts(p1, edges),
        npointstats.pairs.pair_counts(p2, edges),
    )
