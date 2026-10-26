#!/usr/bin/env python3

"""
Property tests for Box and PointPattern geometry.

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
from hypothesis.extra import numpy as hnp
from npst_strategies import boxes, brute_force_min_image, patterns

# ----- local modules -----
from npointstats.box import Box
from npointstats.pattern import PointPattern

# ============================================================
# PROPERTIES
# ============================================================

displacements = hnp.arrays(
    np.float64,
    st.tuples(st.integers(1, 20), st.just(1)),
    elements=st.floats(-4.0, 4.0, allow_nan=False, allow_subnormal=False),
)


@given(boxes())
def test_reciprocal_duality(box: Box) -> None:
    np.testing.assert_allclose(
        box.lattice @ box.reciprocal.T, 2 * np.pi * np.eye(box.dimension), atol=1e-10
    )


@given(patterns())
def test_fractional_round_trip(pattern: PointPattern) -> None:
    box = pattern.box
    x = pattern.positions
    np.testing.assert_allclose(box.to_cartesian(box.to_fractional(x)), x, atol=1e-12)


@given(patterns())
def test_wrap_is_idempotent_and_inside(pattern: PointPattern) -> None:
    box = pattern.box
    shifted = pattern.positions + 3.0 * box.lattice.sum(axis=0)
    once = box.wrap(shifted)
    s = box.to_fractional(once)
    assert np.all(s >= -1e-12)
    assert np.all(s < 1.0 + 1e-12)
    # idempotent up to a lattice translation (points on a face may map to the
    # equivalent opposite face)
    ds = box.to_fractional(box.wrap(once) - once)
    np.testing.assert_allclose(ds, np.rint(ds), atol=1e-9)
    # and the wrapped point is a lattice translate of the original
    dorig = box.to_fractional(once - pattern.positions)
    np.testing.assert_allclose(dorig, np.rint(dorig), atol=1e-9)


@given(boxes(), st.data())
def test_minimum_image_matches_brute_force_below_inscribed_radius(
    box: Box, data: st.DataObject
) -> None:
    n = data.draw(st.integers(1, 20))
    raw = data.draw(
        hnp.arrays(
            np.float64,
            (n, box.dimension),
            elements=st.floats(-3.0, 3.0, allow_nan=False, allow_subnormal=False),
        )
    )
    dx = raw @ box.lattice
    fast = box.minimum_image(dx)
    slow = brute_force_min_image(box, dx)
    fast_len = np.linalg.norm(fast, axis=1)
    slow_len = np.linalg.norm(slow, axis=1)
    # never shorter than the true minimum image, and exact below r_in
    assert np.all(fast_len >= slow_len - 1e-9)
    exact = slow_len < box.inscribed_radius * (1 - 1e-9)
    np.testing.assert_allclose(fast[exact], slow[exact], atol=1e-9)


@given(patterns())
def test_pattern_does_not_alias_input(pattern: PointPattern) -> None:
    raw = np.array(pattern.positions)
    copy = PointPattern(raw, pattern.box)
    raw += 1.0
    np.testing.assert_allclose(copy.positions, pattern.positions)
