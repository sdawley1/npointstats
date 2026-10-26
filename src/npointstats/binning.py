#!/usr/bin/env python3

"""
Radial binning of scattered data and the binned-result container.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from dataclasses import dataclass, field
from typing import Dict

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ============================================================
# RESULT CONTAINER
# ============================================================


@dataclass(frozen=True)
class BinnedResult:
    """
    A function sampled on bins: per-bin means with standard errors.

    Attributes
    ----------
    edges : ndarray, shape (B + 1,)
        Bin edges; bin i is the half-open interval [edges[i], edges[i+1]).
    x_mean : ndarray, shape (B,)
        Mean abscissa of the samples in each bin (NaN for empty bins). For
        discrete data such as |k| shells this is more accurate than the centre.
    mean : ndarray, shape (B,)
        Bin value (NaN for empty bins).
    sem : ndarray, shape (B,)
        Standard error of `mean` (NaN when it cannot be estimated).
    counts : ndarray of int, shape (B,)
        Number of samples (or ensemble members) contributing to each bin.
    metadata : dict
        Free-form provenance (statistic, parameters, N, box, ...).
    """

    edges: NDArray[np.float64]
    x_mean: NDArray[np.float64]
    mean: NDArray[np.float64]
    sem: NDArray[np.float64]
    counts: NDArray[np.int64]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nbins = self.edges.size - 1
        if nbins < 1:
            raise ValueError("BinnedResult needs at least one bin")
        for name in ("x_mean", "mean", "sem", "counts"):
            if getattr(self, name).shape != (nbins,):
                raise ValueError(f"{name} must have shape ({nbins},)")

    @property
    def centers(self) -> NDArray[np.float64]:
        """Bin midpoints."""
        return np.asarray(0.5 * (self.edges[:-1] + self.edges[1:]))

    def to_array(self, nonempty_only: bool = False) -> NDArray[np.float64]:
        """(B, 7) array: lower edge, upper edge, centre, x_mean, mean, sem, count."""
        table = np.column_stack(
            [
                self.edges[:-1],
                self.edges[1:],
                self.centers,
                self.x_mean,
                self.mean,
                self.sem,
                self.counts.astype(np.float64),
            ]
        )
        return np.asarray(table[self.counts > 0] if nonempty_only else table)


# ============================================================
# BINNING
# ============================================================


def uniform_edges(width: float, stop: float, start: float = 0.0) -> NDArray[np.float64]:
    """Edges start, start + width, ... up to the last edge <= stop (at least 1 bin)."""
    if not width > 0:
        raise ValueError(f"bin width must be positive, got {width}")
    if not stop > start:
        raise ValueError(f"stop ({stop}) must exceed start ({start})")
    nbins = max(1, int(np.floor((stop - start) / width + 1e-9)))
    return np.asarray(start + width * np.arange(nbins + 1, dtype=np.float64))


def radial_average(x: ArrayLike, y: ArrayLike, edges: ArrayLike) -> BinnedResult:
    """
    Average `y` over bins of `x`.

    Samples with x outside [edges[0], edges[-1]) are ignored. The input need
    not be sorted. The standard error is the sample standard deviation over
    sqrt(count) (NaN for bins with fewer than two samples).

    Parameters
    ----------
    x, y : array_like, shape (M,)
        Abscissae and values.
    edges : array_like, shape (B + 1,)
        Strictly increasing bin edges.

    Returns
    -------
    BinnedResult

    Examples
    --------
    >>> res = radial_average([0.5, 1.5, 1.7], [1.0, 2.0, 4.0], [0.0, 1.0, 2.0])
    >>> res.mean.tolist(), res.counts.tolist()
    ([1.0, 3.0], [1, 2])
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    e = np.asarray(edges, dtype=np.float64).ravel()
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must have the same length, got {xs.size}, {ys.size}")
    if e.size < 2 or not np.all(np.diff(e) > 0):
        raise ValueError("edges must be strictly increasing with at least 2 entries")
    nbins = e.size - 1
    idx = np.searchsorted(e, xs, side="right") - 1
    ok = (idx >= 0) & (idx < nbins)
    idx, xs, ys = idx[ok], xs[ok], ys[ok]
    counts = np.bincount(idx, minlength=nbins).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(idx, weights=ys, minlength=nbins) / counts
        x_mean = np.bincount(idx, weights=xs, minlength=nbins) / counts
        resid = np.bincount(idx, weights=(ys - mean[idx]) ** 2, minlength=nbins)
        sem = np.sqrt(resid / (counts - 1) / counts)
    sem[counts < 2] = np.nan
    return BinnedResult(edges=e, x_mean=x_mean, mean=mean, sem=sem, counts=counts)
