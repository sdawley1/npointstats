#!/usr/bin/env python3

"""
Pair correlation function g2(r) of a periodic point pattern.

Ordered pairs (i != j) closer than r_max are found with KD-trees (SciPy
`cKDTree.sparse_distance_matrix`) in chunks of query points, so memory stays
bounded instead of the O(N^2) of a full distance matrix, and the work is
proportional to the number of pairs within r_max. Orthorhombic cells use
SciPy's periodic tree; any other cell uses a tree over the 3^d neighbouring
periodic images, which is exact because r_max is limited to the inscribed
radius of the cell (so every pair has exactly one image within range).

Normalisation (exact shell volumes, binomial ideal gas):

    g2(bin) = n_pairs(bin) V / (N (N - 1) [v(r_{i+1}) - v(r_i)]),

with v(r) the volume of a d-ball, so a uniform random (binomial) pattern has
E[g2] = 1 in every bin.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import itertools
import math
from typing import Dict, Optional

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

# ----- local modules -----
from npointstats.binning import BinnedResult, uniform_edges
from npointstats.pattern import PointPattern

# ============================================================
# CONSTANTS
# ============================================================

# Default bin width in units of the mean spacing rho^(-1/d).
DEFAULT_DR_FACTOR = 0.05

# Default r_max in units of the mean spacing (capped by the inscribed radius).
DEFAULT_RMAX_FACTOR = 10.0

# Target number of neighbour pairs materialised per chunk (~24 bytes each).
PAIRS_PER_CHUNK = 2**22

# ============================================================
# GEOMETRY HELPERS
# ============================================================


def ball_volume(r: ArrayLike, d: int) -> NDArray[np.float64]:
    """Volume of a d-dimensional ball of radius r."""
    if d < 1:
        raise ValueError(f"dimension must be >= 1, got {d}")
    radius = np.asarray(r, dtype=np.float64)
    return np.asarray(math.pi ** (d / 2) / math.gamma(1 + d / 2) * radius**d)


def default_dr(pattern: PointPattern) -> float:
    """Default bin width: 0.05 rho^(-1/d)."""
    return float(DEFAULT_DR_FACTOR * pattern.density ** (-1.0 / pattern.dimension))


def default_r_max(pattern: PointPattern) -> float:
    """Default range: min(10 rho^(-1/d), inscribed radius of the box)."""
    spacing = pattern.density ** (-1.0 / pattern.dimension)
    return float(min(DEFAULT_RMAX_FACTOR * spacing, pattern.box.inscribed_radius))


# ============================================================
# PAIR COUNTING
# ============================================================


def _image_points(pattern: PointPattern) -> NDArray[np.float64]:
    """Wrapped positions replicated over the 3^d neighbouring cells."""
    d = pattern.dimension
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=d)), dtype=np.float64)
    offsets = shifts @ pattern.box.lattice
    base = pattern.wrapped()
    return np.asarray((base[None, :, :] + offsets[:, None, :]).reshape(-1, d))


def _chunk_size(pattern: PointPattern, r_max: float) -> int:
    """Query points per chunk so that each chunk yields ~PAIRS_PER_CHUNK pairs."""
    expected = 1.0 + pattern.density * float(ball_volume(r_max, pattern.dimension))
    return int(max(1, min(pattern.n_points, PAIRS_PER_CHUNK // expected)))


def pair_counts(pattern: PointPattern, edges: ArrayLike) -> NDArray[np.int64]:
    """
    Number of ordered pairs (i != j) with minimum-image distance in each bin.

    Work and memory scale with the number of pairs closer than edges[-1],
    N rho v(r_max): cheap for a few neighbour shells, O(N^2) for r_max ~ L/2.

    Parameters
    ----------
    pattern : PointPattern
        Points and periodic cell.
    edges : array_like, shape (B + 1,)
        Increasing bin edges, edges[0] >= 0 and edges[-1] <= inscribed radius.

    Returns
    -------
    ndarray of int, shape (B,)
        Pair counts in the half-open bins [edges[i], edges[i+1]).
    """
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 1 or e.size < 2 or not np.all(np.diff(e) > 0) or e[0] < 0:
        raise ValueError("edges must be non-negative and strictly increasing")
    r_in = pattern.box.inscribed_radius
    if e[-1] > r_in * (1 + 1e-12):
        raise ValueError(
            f"r_max={e[-1]:g} exceeds the inscribed radius {r_in:g} of the box; "
            "minimum-image pair statistics are not defined beyond it"
        )
    if pattern.box.is_orthorhombic:
        boxsize = np.diag(pattern.box.lattice).copy()
        # build from fractional coordinates so values lie in [0, L) exactly
        x = pattern.fractional() * boxsize
        x[x >= boxsize] = 0.0
        target = cKDTree(x, boxsize=boxsize)
    else:
        boxsize = None
        x = pattern.wrapped()
        target = cKDTree(_image_points(pattern))
    nbins = e.size - 1
    counts = np.zeros(nbins, dtype=np.int64)
    chunk = _chunk_size(pattern, float(e[-1]))
    for start in range(0, pattern.n_points, chunk):
        query = cKDTree(x[start : start + chunk], boxsize=boxsize)
        dist = query.sparse_distance_matrix(target, e[-1], output_type="ndarray")["v"]
        idx = np.searchsorted(e, dist, side="right") - 1
        counts += np.bincount(idx[(idx >= 0) & (idx < nbins)], minlength=nbins)
    if e[0] == 0.0:
        counts[0] -= pattern.n_points  # remove the i == j self pairs
    if np.any(counts < 0):
        raise RuntimeError("negative pair count; please report this as a bug")
    return counts


# ============================================================
# PUBLIC API
# ============================================================


def pair_correlation(
    pattern: PointPattern,
    dr: Optional[float] = None,
    r_max: Optional[float] = None,
) -> BinnedResult:
    """
    Pair correlation function g2(r) on uniform bins [j dr, (j + 1) dr).

    Parameters
    ----------
    pattern : PointPattern
        Points and periodic cell (any shape), N >= 2.
    dr : float, optional
        Bin width in the units of the positions. Default 0.05 rho^(-1/d).
    r_max : float, optional
        Largest radius. Default min(10 rho^(-1/d), inscribed radius); upper
        limit the inscribed radius (half the smallest face separation). Cost
        grows like N rho v(r_max), i.e. O(N^2) for r_max ~ L/2.

    Returns
    -------
    BinnedResult
        `mean` is g2, `sem` its Poisson counting error, `counts` the number of
        unordered pairs per bin, `x_mean` the bin centres.

    Examples
    --------
    >>> import numpy as np
    >>> from npointstats.box import Box
    >>> grid = np.stack(np.meshgrid(np.arange(4), np.arange(4)), -1).reshape(-1, 2)
    >>> g2 = pair_correlation(PointPattern(grid, Box.cubic(4.0, 2)), dr=0.25)
    >>> g2.counts.tolist()  # 16 points x 4 neighbours / 2 at r = 1 and r = sqrt 2
    [0, 0, 0, 0, 32, 32, 0, 0]
    """
    if pattern.n_points < 2:
        raise ValueError("g2 needs at least two points")
    width = default_dr(pattern) if dr is None else float(dr)
    r_in = pattern.box.inscribed_radius
    stop = default_r_max(pattern) if r_max is None else float(r_max)
    if stop > r_in * (1 + 1e-12):
        raise ValueError(f"r_max={stop:g} exceeds the inscribed radius {r_in:g}")
    edges = uniform_edges(width, stop)
    ordered = pair_counts(pattern, edges)
    n, d = pattern.n_points, pattern.dimension
    shell = np.diff(ball_volume(edges, d))
    norm = n * (n - 1) * shell / pattern.box.volume
    meta: Dict[str, object] = {
        "statistic": "pair_correlation",
        "dr": width,
        "r_max": float(edges[-1]),
        "n_points": n,
        "dimension": d,
    }
    return BinnedResult(
        edges=edges,
        x_mean=0.5 * (edges[:-1] + edges[1:]),
        mean=ordered / norm,
        sem=np.sqrt(2.0 * ordered) / norm,
        counts=ordered // 2,
        metadata=meta,
    )
