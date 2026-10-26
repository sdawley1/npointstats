#!/usr/bin/env python3

"""
Property tests for the structure factor.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- third party -----
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from npst_strategies import pattern_and_shift, patterns

# ----- local modules -----
from npointstats.pattern import PointPattern
from npointstats.structure_factor import StructureFactor, structure_factor

# ============================================================
# HELPERS
# ============================================================


def small_sk(pattern: PointPattern, method: str = "nufft") -> StructureFactor:
    """S(k) up to roughly five shells of the cell, cheap for property tests."""
    k_max = 5.0 * 2 * np.pi / pattern.box.inscribed_radius / 2
    return structure_factor(pattern, k_max=k_max, method=method)


# ============================================================
# PROPERTIES
# ============================================================


@given(patterns())
def test_nufft_matches_direct_sum(pattern: PointPattern) -> None:
    a = small_sk(pattern, "nufft")
    b = small_sk(pattern, "direct")
    np.testing.assert_array_equal(a.k_vectors, b.k_vectors)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-8, atol=1e-8)


@given(patterns())
def test_bounds_and_inversion_symmetry(pattern: PointPattern) -> None:
    sk = small_sk(pattern)
    assert np.all(sk.values >= 0.0)
    assert np.all(sk.values <= pattern.n_points * (1 + 1e-9))
    assert np.all(sk.k <= sk.k_max * (1 + 1e-12))
    # S(k) = S(-k): match each vector with its negative
    lookup = {tuple(np.round(k, 9)): v for k, v in zip(sk.k_vectors, sk.values, strict=True)}
    for k, v in zip(sk.k_vectors, sk.values, strict=True):
        np.testing.assert_allclose(lookup[tuple(np.round(-k, 9))], v, atol=1e-8)


@given(pattern_and_shift())
def test_translation_invariance(case: tuple[PointPattern, np.ndarray]) -> None:
    pattern, shift = case
    moved = PointPattern(pattern.positions + shift, pattern.box)
    np.testing.assert_allclose(
        small_sk(pattern).values, small_sk(moved).values, rtol=1e-8, atol=1e-8
    )


@given(patterns(min_n=2), st.randoms(use_true_random=False), st.data())
def test_permutation_and_lattice_translation_invariance(
    pattern: PointPattern, rand: "object", data: st.DataObject
) -> None:
    order = list(range(pattern.n_points))
    rand.shuffle(order)  # type: ignore[attr-defined]
    n = data.draw(
        st.lists(
            st.integers(-2, 2), min_size=pattern.dimension, max_size=pattern.dimension
        )
    )
    pos = pattern.positions[order].copy()
    pos[0] += np.array(n, dtype=np.float64) @ pattern.box.lattice
    other = PointPattern(pos, pattern.box)
    np.testing.assert_allclose(
        small_sk(pattern).values, small_sk(other).values, rtol=1e-8, atol=1e-8
    )


@given(patterns(), st.floats(0.2, 5.0))
def test_scale_invariance(pattern: PointPattern, c: float) -> None:
    scaled = PointPattern(c * pattern.positions, c * pattern.box.lattice)
    a = small_sk(pattern)
    b = structure_factor(scaled, k_max=a.k_max / c)
    # order within a degenerate shell is unspecified: match vectors by index
    ia = np.rint(a.k_vectors @ pattern.box.lattice.T / (2 * np.pi)).astype(int)
    ib = np.rint(b.k_vectors @ scaled.box.lattice.T / (2 * np.pi)).astype(int)
    sa = {tuple(n): v for n, v in zip(ia, a.values, strict=True)}
    sb = {tuple(n): v for n, v in zip(ib, b.values, strict=True)}
    # shells exactly at k_max may differ by rounding; compare the common set
    common = set(sa) & set(sb)
    assert len(common) >= 0.9 * max(len(sa), len(sb))
    for n in common:
        np.testing.assert_allclose(sb[n], sa[n], rtol=1e-8, atol=1e-8)
