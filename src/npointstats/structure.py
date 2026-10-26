#!/usr/bin/env python3

"""
Structure factor S(k) of a periodic point pattern.

For N points in a periodic cell, S is defined on the reciprocal lattice,

    S(k) = |rho(k)|^2 / N,    rho(k) = sum_j exp(i k . r_j),    k != 0,

with k = n @ B for integer n and reciprocal lattice vectors B (rows). Since
k . r_j = 2 pi n . s_j for fractional coordinates s_j, every reciprocal
vector inside a sphere |k| <= k_max is obtained exactly from one type-1
non-uniform FFT (FINUFFT) on 2 pi s_j, in O(N + M log M) time. A chunked
direct sum gives the same numbers in O(N M) time with bounded memory and
serves as a reference.

The full, inversion-symmetric set of reciprocal vectors with 0 < |k| <= k_max
is used, so every shell |k| <= k_max is complete for any cell shape.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from dataclasses import dataclass
from typing import Callable, Dict, Optional

# ----- third party -----
import finufft
import numpy as np
from numpy.typing import NDArray

# ----- local modules -----
from npointstats.binning import BinnedResult, radial_average
from npointstats.box import Box
from npointstats.pattern import PointPattern

# ============================================================
# CONSTANTS
# ============================================================

# Default k_max in units of 2 pi rho^(1/d) (about 3 nearest-neighbour peaks).
DEFAULT_K_MAX_FACTOR = 3.0

# Upper bound on the number of complex phases held at once by the direct sum.
DIRECT_CHUNK_ELEMENTS = 2**22

# Hard cap on the size of the FINUFFT mode grid, to fail fast on typos in k_max.
MAX_MODES = 2**28

METHODS = ("nufft", "direct")

_NUFFT_TYPE1: Dict[int, Callable[..., NDArray[np.complex128]]] = {
    1: finufft.nufft1d1,
    2: finufft.nufft2d1,
    3: finufft.nufft3d1,
}

# ============================================================
# RESULT CONTAINER
# ============================================================


@dataclass(frozen=True)
class StructureFactor:
    """
    S(k) on every reciprocal-lattice vector with 0 < |k| <= k_max.

    Attributes
    ----------
    k_vectors : ndarray, shape (M, d)
        Wave vectors, sorted by magnitude.
    k : ndarray, shape (M,)
        Magnitudes |k|.
    values : ndarray, shape (M,)
        S(k).
    k_max : float
        Cutoff used; all shells with |k| <= k_max are complete.
    n_points : int
        Number of points N.
    box : Box
        Periodic cell of the pattern.
    """

    k_vectors: NDArray[np.float64]
    k: NDArray[np.float64]
    values: NDArray[np.float64]
    k_max: float
    n_points: int
    box: Box

    def radial_average(
        self, dk: Optional[float] = None, k_max: Optional[float] = None
    ) -> BinnedResult:
        """
        Average S over spherical shells of width `dk`.

        Bins are centred on multiples of `dk` (edges at (j + 1/2) dk), and only
        bins lying entirely below `k_max` are kept, so each bin averages over
        complete shells. `dk` defaults to the smallest reciprocal-vector norm
        (one shell spacing of a cubic cell).
        """
        width = default_dk(self.box) if dk is None else float(dk)
        cut = self.k_max if k_max is None else min(float(k_max), self.k_max)
        edges = shell_edges(width, cut)
        result = radial_average(self.k, self.values, edges)
        meta: Dict[str, object] = {
            "statistic": "structure_factor",
            "dk": width,
            "k_max": cut,
            "n_points": self.n_points,
            "dimension": self.box.dimension,
        }
        return BinnedResult(
            edges=result.edges,
            x_mean=result.x_mean,
            mean=result.mean,
            sem=result.sem,
            counts=result.counts,
            metadata=meta,
        )


# ============================================================
# WAVE VECTORS
# ============================================================


def default_k_max(pattern: PointPattern) -> float:
    """Default cutoff: 3 x 2 pi rho^(1/d)."""
    return float(
        DEFAULT_K_MAX_FACTOR
        * 2.0
        * np.pi
        * pattern.density ** (1.0 / pattern.dimension)
    )


def default_dk(box: Box) -> float:
    """Default shell width: the smallest reciprocal lattice vector norm."""
    return float(np.min(np.linalg.norm(box.reciprocal, axis=1)))


def shell_edges(dk: float, k_max: float) -> NDArray[np.float64]:
    """Edges (j + 1/2) dk, j = 0, 1, ..., with the last edge <= k_max."""
    if not dk > 0:
        raise ValueError(f"dk must be positive, got {dk}")
    nbins = int(np.floor((k_max - 0.5 * dk) / dk + 1e-9))
    if nbins < 1:
        raise ValueError(f"k_max={k_max:g} is smaller than 1.5 dk={1.5 * dk:g}")
    return np.asarray(dk * (np.arange(nbins + 1, dtype=np.float64) + 0.5))


def _index_bounds(box: Box, k_max: float) -> NDArray[np.int64]:
    """Largest |n_i| that can satisfy |n @ B| <= k_max: k_max |a_i| / (2 pi)."""
    bound = np.floor(k_max * np.linalg.norm(box.lattice, axis=1) / (2.0 * np.pi) + 1e-9)
    return bound.astype(np.int64)


def wavevector_indices(box: Box, k_max: float) -> NDArray[np.int64]:
    """
    Integer indices n of all reciprocal vectors k = n @ B with 0 < |k| <= k_max.

    Returned sorted by |k|; the set is closed under n -> -n.
    """
    if not k_max > 0:
        raise ValueError(f"k_max must be positive, got {k_max}")
    bounds = _index_bounds(box, k_max)
    n_modes = int(np.prod(2 * bounds + 1))
    if n_modes > MAX_MODES:
        raise ValueError(f"k_max={k_max:g} needs {n_modes} modes (> {MAX_MODES})")
    axes = [np.arange(-b, b + 1, dtype=np.int64) for b in bounds]
    grid = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    kmag = np.linalg.norm(grid @ box.reciprocal, axis=1)
    keep = (kmag <= k_max * (1 + 1e-12)) & np.any(grid != 0, axis=1)
    order = np.argsort(kmag[keep], kind="stable")
    return np.asarray(grid[keep][order])


# ============================================================
# BACKENDS
# ============================================================


def _rho_nufft(
    pattern: PointPattern, indices: NDArray[np.int64], eps: float, nthreads: int
) -> NDArray[np.complex128]:
    """Collective density rho(k) on `indices` via one type-1 FINUFFT."""
    d = pattern.dimension
    if d not in _NUFFT_TYPE1:
        raise ValueError(f"method='nufft' supports d = 1, 2, 3, got d = {d}")
    bounds = np.max(np.abs(indices), axis=0)
    n_modes = tuple(int(2 * b + 1) for b in bounds)
    phases = 2.0 * np.pi * pattern.fractional()
    coords = [np.ascontiguousarray(phases[:, i]) for i in range(d)]
    weights = np.ones(pattern.n_points, dtype=np.complex128)
    grid = _NUFFT_TYPE1[d](
        *coords,
        weights,
        n_modes if d > 1 else n_modes[0],
        eps=eps,
        isign=1,
        modeord=1,
        nthreads=nthreads,
    )
    # modeord=1 stores mode n at position n mod n_modes along each axis
    pos = tuple(np.mod(indices[:, i], n_modes[i]) for i in range(d))
    return np.asarray(np.reshape(grid, n_modes)[pos])


def _rho_direct(
    pattern: PointPattern, k_vectors: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Collective density by explicit summation, chunked to bound memory."""
    x = pattern.wrapped()
    out = np.empty(k_vectors.shape[0], dtype=np.complex128)
    chunk = max(1, DIRECT_CHUNK_ELEMENTS // pattern.n_points)
    for start in range(0, k_vectors.shape[0], chunk):
        phase = k_vectors[start : start + chunk] @ x.T
        out[start : start + chunk] = np.exp(1j * phase).sum(axis=1)
    return out


# ============================================================
# PUBLIC API
# ============================================================


def structure_factor(
    pattern: PointPattern,
    k_max: Optional[float] = None,
    method: str = "nufft",
    eps: float = 1e-12,
    nthreads: int = 1,
) -> StructureFactor:
    """
    Structure factor on all reciprocal-lattice vectors with 0 < |k| <= k_max.

    Parameters
    ----------
    pattern : PointPattern
        Points and periodic cell (any shape).
    k_max : float, optional
        Largest |k| (same length units as the positions, inverted). Defaults to
        3 x 2 pi rho^(1/d).
    method : {"nufft", "direct"}
        FINUFFT type-1 transform (fast, d <= 3) or chunked direct summation.
    eps : float
        FINUFFT relative tolerance (ignored by "direct").
    nthreads : int
        FINUFFT threads. The default of 1 is deliberate: FINUFFT's own default
        (all cores) is orders of magnitude slower on shared many-core nodes for
        small and medium problems. Parallelise ensembles across members instead
        (``Ensemble(..., jobs=...)``), or raise this on a dedicated node.

    Returns
    -------
    StructureFactor

    Examples
    --------
    >>> lattice = PointPattern([[0.0, 0.0], [0.5, 0.5]], Box.cubic(1.0, 2))
    >>> sk = structure_factor(lattice, k_max=9.0)
    >>> sk.values.round(9).tolist()  # 4 vectors |k| = 2 pi, 4 vectors 2 pi sqrt 2
    [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    cut = default_k_max(pattern) if k_max is None else float(k_max)
    indices = wavevector_indices(pattern.box, cut)
    if indices.shape[0] == 0:
        raise ValueError(f"no reciprocal vectors with |k| <= k_max={cut:g}")
    k_vectors = indices @ pattern.box.reciprocal
    if method == "nufft":
        if nthreads < 1:
            raise ValueError(f"nthreads must be >= 1, got {nthreads}")
        rho = _rho_nufft(pattern, indices, eps, nthreads)
    else:
        rho = _rho_direct(pattern, k_vectors)
    values = (rho.real**2 + rho.imag**2) / pattern.n_points
    return StructureFactor(
        k_vectors=k_vectors,
        k=np.linalg.norm(k_vectors, axis=1),
        values=values,
        k_max=cut,
        n_points=pattern.n_points,
        box=pattern.box,
    )
