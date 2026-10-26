#!/usr/bin/env python3

"""
Local number variance sigma^2(R) of a periodic point pattern.

Two estimators:

- `number_variance`: spherical windows of radius R centred at uniformly random
  points of the cell; sigma^2(R) is the sample variance of the window counts.
  All radii are counted in one chunked KD-tree pass.
- `number_variance_from_sk`: the exact torus identity

      sigma^2(R) = rho v(R) (1/V) sum_{k != 0} S(k) alpha(k; R),
      alpha(k; R) = 2^d pi^(d/2) Gamma(1 + d/2) J_{d/2}(kR)^2 / k^d,

  (Torquato & Stillinger, Phys. Rev. E 68, 041113 (2003)) evaluated on the
  reciprocal vectors of a `StructureFactor`. Since (1/V) sum_{k != 0}
  alpha = 1 - v/V (Parseval), vectors beyond k_max are included exactly under
  the assumption S = 1 there:

      sigma^2(R) = rho v(R) [ (1/V) sum_{0<|k|<=k_max} (S - 1) alpha + 1 - v/V ].

Both require R <= the inscribed radius of the cell (windows must not overlap
their own periodic images).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree
from scipy.special import jv

# ----- local modules -----
from npointstats.pairs import PAIRS_PER_CHUNK, _image_points, ball_volume
from npointstats.pattern import PointPattern
from npointstats.structure import StructureFactor

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_N_WINDOWS = 5000
DEFAULT_N_RADII = 40
# Default largest radius in units of the mean spacing (capped at r_in).
DEFAULT_RMAX_FACTOR = 5.0

# ============================================================
# RESULT CONTAINER
# ============================================================


@dataclass(frozen=True)
class NumberVariance:
    """
    sigma^2(R) sampled at radii R.

    Attributes
    ----------
    radii : ndarray, shape (R,)
    variance : ndarray, shape (R,)
        sigma^2(R).
    sem : ndarray, shape (R,)
        Standard error of `variance` (NaN when not estimated).
    mean_count : ndarray, shape (R,)
        Mean number of points per window (rho v(R) in expectation).
    metadata : dict
    """

    radii: NDArray[np.float64]
    variance: NDArray[np.float64]
    sem: NDArray[np.float64]
    mean_count: NDArray[np.float64]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = self.radii.shape
        for name in ("variance", "sem", "mean_count"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")

    def to_array(self) -> NDArray[np.float64]:
        """(R, 4) array: radius, variance, sem, mean count."""
        return np.column_stack([self.radii, self.variance, self.sem, self.mean_count])


# ============================================================
# HELPERS
# ============================================================


def default_radii(
    pattern: PointPattern, n: int = DEFAULT_N_RADII
) -> NDArray[np.float64]:
    """n radii up to min(5 rho^(-1/d), inscribed radius), evenly spaced."""
    spacing = pattern.density ** (-1.0 / pattern.dimension)
    r_max = min(DEFAULT_RMAX_FACTOR * spacing, pattern.box.inscribed_radius)
    return np.asarray(np.linspace(r_max / n, r_max, n))


def _check_radii(pattern: PointPattern, radii: ArrayLike) -> NDArray[np.float64]:
    r = np.asarray(radii, dtype=np.float64).ravel()
    if r.size == 0 or not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("radii must be a non-empty array of positive numbers")
    if not np.all(np.diff(r) > 0):
        raise ValueError("radii must be strictly increasing")
    r_in = pattern.box.inscribed_radius
    if r[-1] > r_in * (1 + 1e-12):
        raise ValueError(f"largest radius {r[-1]:g} exceeds inscribed radius {r_in:g}")
    return r


def window_counts(
    pattern: PointPattern, centers: ArrayLike, radii: ArrayLike
) -> NDArray[np.int64]:
    """
    Number of points strictly inside each window, for every radius.

    Parameters
    ----------
    pattern : PointPattern
    centers : array_like, shape (W, d)
        Window centres (Cartesian).
    radii : array_like, shape (R,)
        Increasing radii, at most the inscribed radius.

    Returns
    -------
    ndarray of int, shape (W, R)
    """
    r = _check_radii(pattern, radii)
    raw = np.asarray(centers, dtype=np.float64).reshape(-1, pattern.dimension)
    sc = pattern.box.to_reduced_fractional(raw)
    sc = sc - np.floor(sc)
    sc[sc >= 1.0] = 0.0
    c = sc @ pattern.box.reduced_lattice
    if pattern.box.is_orthorhombic:
        boxsize = np.diag(pattern.box.lattice).copy()
        pts = pattern.fractional() * boxsize
        pts[pts >= boxsize] = 0.0
        c = pattern.box.to_fractional(c) % 1.0 * boxsize
        c[c >= boxsize] = 0.0
        target = cKDTree(pts, boxsize=boxsize)
    else:
        boxsize = None
        target = cKDTree(_image_points(pattern))
    n_windows, nr = c.shape[0], r.size
    hist = np.zeros(n_windows * (nr + 1), dtype=np.int64)
    expected = 1.0 + pattern.density * float(ball_volume(r[-1], pattern.dimension))
    chunk = int(max(1, min(n_windows, PAIRS_PER_CHUNK // expected)))
    for start in range(0, n_windows, chunk):
        query = cKDTree(c[start : start + chunk], boxsize=boxsize)
        pairs = query.sparse_distance_matrix(target, r[-1], output_type="ndarray")
        b = np.searchsorted(r, pairs["v"], side="right")  # radii <= d
        hist += np.bincount((pairs["i"] + start) * (nr + 1) + b, minlength=hist.size)
    counts = np.cumsum(hist.reshape(n_windows, nr + 1), axis=1)[:, :nr]
    return np.asarray(counts)


# ============================================================
# ESTIMATORS
# ============================================================


def number_variance(
    pattern: PointPattern,
    radii: Optional[ArrayLike] = None,
    n_windows: int = DEFAULT_N_WINDOWS,
    seed: Optional[int] = None,
) -> NumberVariance:
    """
    sigma^2(R) from randomly placed spherical windows.

    Parameters
    ----------
    pattern : PointPattern
    radii : array_like, optional
        Increasing window radii (<= inscribed radius). Default: 40 radii up to
        min(5 rho^(-1/d), inscribed radius).
    n_windows : int
        Number of random window centres (shared by all radii).
    seed : int, optional
        Seed for `numpy.random.default_rng`; fix it for reproducible results.

    Returns
    -------
    NumberVariance
        `sem` is the large-sample standard error sqrt((m4 - s^4) / W).

    Examples
    --------
    >>> from npointstats.box import Box
    >>> import numpy as np
    >>> lattice = PointPattern(np.arange(10.0)[:, None], Box.cubic(10.0, 1))
    >>> nv = number_variance(lattice, radii=[0.5, 1.0], n_windows=4000, seed=0)
    >>> bool(abs(nv.variance[0] - 0.0) < 1e-12)  # each unit window holds 1 point
    True
    """
    if n_windows < 2:
        raise ValueError(f"n_windows must be >= 2, got {n_windows}")
    r = default_radii(pattern) if radii is None else _check_radii(pattern, radii)
    rng = np.random.default_rng(seed)
    centers = pattern.box.to_cartesian(rng.random((n_windows, pattern.dimension)))
    counts = window_counts(pattern, centers, r).astype(np.float64)
    mean = counts.mean(axis=0)
    dev = counts - mean
    var = np.sum(dev**2, axis=0) / (n_windows - 1)
    m4 = np.mean(dev**4, axis=0)
    sem = np.sqrt(np.maximum(m4 - var**2, 0.0) / n_windows)
    meta: Dict[str, object] = {
        "statistic": "number_variance",
        "estimator": "windows",
        "n_windows": n_windows,
        "seed": seed,
        "n_points": pattern.n_points,
        "dimension": pattern.dimension,
    }
    return NumberVariance(
        radii=r, variance=var, sem=sem, mean_count=mean, metadata=meta
    )


def alpha_tilde(k: ArrayLike, radius: float, d: int) -> NDArray[np.float64]:
    """
    Fourier transform of the scaled intersection volume, alpha(k; R).

    alpha(k; R) = |w(k)|^2 / v(R) for the indicator w of a d-ball of radius R;
    alpha(0; R) = v(R).
    """
    kk = np.asarray(k, dtype=np.float64)
    pref = 2.0**d * math.pi ** (d / 2) * math.gamma(1 + d / 2)
    out = np.empty_like(kk)
    small = kk * radius < 1e-8
    ks = kk[~small]
    out[~small] = pref * jv(d / 2, ks * radius) ** 2 / ks**d
    out[small] = float(ball_volume(radius, d))
    return out


def number_variance_from_sk(
    sk: StructureFactor, radii: ArrayLike, ideal_tail: bool = True
) -> NumberVariance:
    """
    sigma^2(R) from the structure factor (exact torus identity).

    Parameters
    ----------
    sk : StructureFactor
        S(k) on all reciprocal vectors with 0 < |k| <= k_max.
    radii : array_like
        Window radii (<= inscribed radius of the cell).
    ideal_tail : bool, default True
        Include vectors beyond k_max assuming S = 1 there (exact for that
        assumption). If False, the sum is simply truncated at k_max.

    Returns
    -------
    NumberVariance
        `sem` is NaN (deterministic estimate); `mean_count` is rho v(R).
    """
    box = sk.box
    d = box.dimension
    dummy = PointPattern(np.zeros((1, d)), box)
    r = _check_radii(dummy, radii)
    rho = sk.n_points / box.volume
    var = np.empty_like(r)
    for i, radius in enumerate(r):
        a = alpha_tilde(sk.k, float(radius), d)
        v = float(ball_volume(radius, d))
        if ideal_tail:
            inner = np.sum((sk.values - 1.0) * a) / box.volume + 1.0 - v / box.volume
        else:
            inner = np.sum(sk.values * a) / box.volume
        var[i] = rho * v * inner
    meta: Dict[str, object] = {
        "statistic": "number_variance",
        "estimator": "structure_factor",
        "k_max": sk.k_max,
        "ideal_tail": ideal_tail,
        "n_points": sk.n_points,
        "dimension": d,
    }
    return NumberVariance(
        radii=r,
        variance=var,
        sem=np.full_like(r, np.nan),
        mean_count=rho * ball_volume(r, d),
        metadata=meta,
    )
