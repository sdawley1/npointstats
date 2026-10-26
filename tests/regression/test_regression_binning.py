#!/usr/bin/env python3

"""
Regression and property tests for radial binning (D-002).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import random

# ----- third party -----
import numpy as np
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ----- local modules -----
from npointstats.binning import radial_average

# ============================================================
# STRATEGIES
# ============================================================

values = st.floats(-100.0, 100.0, allow_nan=False, allow_subnormal=False)


@st.composite
def xy_edges(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = draw(st.integers(1, 60))
    x = draw(hnp.arrays(np.float64, n, elements=st.floats(0.0, 10.0)))
    y = draw(hnp.arrays(np.float64, n, elements=values))
    nb = draw(st.integers(1, 12))
    edges = np.linspace(0.0, 10.0 + 1e-9, nb + 1)
    return x, y, edges


# ============================================================
# D-002
# ============================================================


def test_regression_D002_last_bin_kept_and_unsorted_input() -> None:
    """D-002: the largest-x bin is reported and input order does not matter."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    res = radial_average(x, y, np.array([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(res.mean, [1.0, 2.5])
    np.testing.assert_array_equal(res.counts, [1, 2])
    # edge: reversed (unsorted) input gives the same bins
    rev = radial_average(x[::-1], y[::-1], np.array([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(rev.mean, res.mean)
    # edge: a single bin holding everything
    one = radial_average(x, y, np.array([0.0, 3.0]))
    np.testing.assert_allclose(one.mean, [2.0])


@given(xy_edges())
@example(
    (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 3.0]))
)
def test_regression_D002_counts_and_means_conserved(
    case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """D-002 property: every in-range point lands in exactly one bin."""
    x, y, edges = case
    res = radial_average(x, y, edges)
    inside = (x >= edges[0]) & (x < edges[-1])
    assert res.counts.sum() == inside.sum()
    filled = res.counts > 0
    total = np.sum(res.mean[filled] * res.counts[filled])
    scale = max(1.0, float(np.abs(y).sum()))
    np.testing.assert_allclose(total, y[inside].sum(), rtol=1e-9, atol=1e-9 * scale)


@given(xy_edges(), st.randoms(use_true_random=False))
def test_binning_permutation_invariant(
    case: tuple[np.ndarray, np.ndarray, np.ndarray],
    rand: random.Random,
) -> None:
    x, y, edges = case
    order = list(range(x.size))
    rand.shuffle(order)
    a = radial_average(x, y, edges)
    b = radial_average(x[order], y[order], edges)
    np.testing.assert_array_equal(a.counts, b.counts)
    np.testing.assert_allclose(a.mean, b.mean, rtol=1e-12, atol=1e-10, equal_nan=True)


def test_binned_result_table_and_validation() -> None:
    res = radial_average([0.5, 2.5], [1.0, 3.0], [0.0, 1.0, 2.0, 3.0])
    table = res.to_array()
    assert table.shape == (3, 7)
    np.testing.assert_allclose(res.centers, [0.5, 1.5, 2.5])
    assert res.to_array(nonempty_only=True).shape == (2, 7)
    assert np.isnan(res.mean[1])
    assert np.isnan(res.sem[0])


def test_binning_input_errors() -> None:
    import pytest

    from npointstats.binning import BinnedResult, uniform_edges

    with pytest.raises(ValueError, match="same length"):
        radial_average([1.0], [1.0, 2.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        radial_average([1.0], [1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        uniform_edges(0.0, 1.0)
    with pytest.raises(ValueError, match="exceed"):
        uniform_edges(0.1, -1.0)
    np.testing.assert_allclose(uniform_edges(0.25, 1.0), [0, 0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError, match="at least one bin"):
        BinnedResult(
            np.array([0.0]), *(np.array([]),) * 3, np.array([], dtype=np.int64)
        )
    with pytest.raises(ValueError, match="shape"):
        BinnedResult(
            np.array([0.0, 1.0]),
            np.zeros(2),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1, int),
        )


@st.composite
def edges_and_values(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    nb = draw(st.integers(1, 50))
    start = draw(st.floats(-5.0, 5.0))
    width = draw(st.floats(1e-3, 2.0))
    if draw(st.booleans()):
        edges = start + width * np.arange(nb + 1, dtype=float)
    else:
        steps = draw(hnp.arrays(np.float64, nb, elements=st.floats(1e-3, 2.0)))
        edges = start + np.concatenate([[0.0], np.cumsum(steps)])
    picks = draw(
        hnp.arrays(np.int64, draw(st.integers(0, 30)), elements=st.integers(0, nb))
    )
    noise = draw(hnp.arrays(np.float64, picks.size, elements=st.floats(-3.0, 3.0)))
    # values exactly on edges, on their float neighbours, and in between
    on_edge = edges[picks]
    near = np.nextafter(on_edge, on_edge + np.sign(noise + 1e-300))
    spread = start + noise * width * nb
    return edges, np.concatenate([on_edge, near, spread])


@given(edges_and_values())
@example(
    (
        np.array([0.0, 0.1, 0.2, 0.3]),
        np.array([0.0, 0.1, 0.2, 0.3, 0.30000000000000004]),
    )
)
def test_bin_indices_match_searchsorted(case: tuple[np.ndarray, np.ndarray]) -> None:
    from npointstats.binning import bin_indices

    edges, values = case
    want = np.searchsorted(edges, values, side="right") - 1
    want[(want < 0) | (want >= edges.size - 1)] = -1
    np.testing.assert_array_equal(bin_indices(edges, values), want)
