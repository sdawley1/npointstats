#!/usr/bin/env python3

"""
Hypothesis strategies shared by the npointstats test suite.

Boxes are drawn as lower-triangular sheared cells with bounded shear, so
they are always well conditioned; any lattice is a rotation of such a cell.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Tuple

# ----- third party -----
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.typing import NDArray

# ----- local modules -----
from npointstats.box import Box
from npointstats.pattern import PointPattern

# ============================================================
# STRATEGIES
# ============================================================

unit_float = st.floats(
    0.0, 1.0, exclude_max=True, allow_nan=False, allow_subnormal=False
)
length = st.floats(0.5, 3.0, allow_nan=False, allow_subnormal=False)
shear = st.floats(-0.6, 0.6, allow_nan=False, allow_subnormal=False)


@st.composite
def boxes(
    draw: st.DrawFn, dimension: int | None = None, orthorhombic: bool = False
) -> Box:
    """Draw a well-conditioned periodic box (orthorhombic or sheared)."""
    d = dimension if dimension is not None else draw(st.integers(1, 3))
    lengths = np.array([draw(length) for _ in range(d)])
    lattice = np.diag(lengths)
    if not orthorhombic:
        for i in range(d):
            for j in range(i):
                lattice[i, j] = draw(shear) * lengths[i]
    return Box(lattice)


@st.composite
def patterns(
    draw: st.DrawFn,
    dimension: int | None = None,
    min_n: int = 1,
    max_n: int = 30,
    orthorhombic: bool = False,
) -> PointPattern:
    """Draw a point pattern with uniform fractional coordinates in a random box."""
    box = draw(boxes(dimension=dimension, orthorhombic=orthorhombic))
    n = draw(st.integers(min_n, max_n))
    s = draw(hnp.arrays(np.float64, (n, box.dimension), elements=unit_float))
    return PointPattern(box.to_cartesian(s), box)


@st.composite
def pattern_and_shift(
    draw: st.DrawFn, dimension: int | None = None, max_n: int = 30
) -> Tuple[PointPattern, NDArray[np.float64]]:
    """Draw a pattern plus a Cartesian translation vector of the same dimension."""
    pattern = draw(patterns(dimension=dimension, max_n=max_n))
    s = draw(hnp.arrays(np.float64, pattern.dimension, elements=unit_float))
    return pattern, np.asarray(pattern.box.to_cartesian(s))


def brute_force_min_image(box: Box, dx: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shortest image of each displacement by searching shifts in {-4..4}^d."""
    d = box.dimension
    grids = np.meshgrid(*[np.arange(-4, 5)] * d, indexing="ij")
    shifts = np.stack([g.ravel() for g in grids], axis=1) @ box.lattice
    cand = dx[:, None, :] + shifts[None, :, :]
    best = np.argmin(np.linalg.norm(cand, axis=2), axis=1)
    return np.asarray(cand[np.arange(dx.shape[0]), best])
