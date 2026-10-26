#!/usr/bin/env python3

"""
Ensemble averages of n-point statistics over many point patterns.

Every member is binned onto the same edges (or radii) before averaging, so
members may have different boxes and particle numbers. The ensemble standard
error is the member-to-member standard deviation over sqrt(members); for a
single member the within-member error is returned instead.

Members can be processed in parallel with `jobs > 1` (processes); results are
identical to the serial run because members are reduced in input order and
random seeds are derived per member from one `numpy.random.SeedSequence`.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ----- local modules -----
from npointstats.binning import BinnedResult
from npointstats.io import FORMATS
from npointstats.number_variance import NumberVariance, default_radii, number_variance
from npointstats.pair_correlation import default_dr, default_r_max, pair_correlation
from npointstats.pattern import PointPattern
from npointstats.structure_factor import default_dk, default_k_max, structure_factor

# ============================================================
# WORKERS (module level so they pickle)
# ============================================================


def _sf_worker(args: Tuple[PointPattern, Dict[str, Any]]) -> BinnedResult:
    pattern, kw = args
    sk = structure_factor(
        pattern, k_max=kw["k_max"], method=kw["method"], nthreads=kw["nthreads"]
    )
    return sk.radial_average(dk=kw["dk"])


def _g2_worker(args: Tuple[PointPattern, Dict[str, Any]]) -> BinnedResult:
    pattern, kw = args
    return pair_correlation(pattern, dr=kw["dr"], r_max=kw["r_max"])


def _nv_worker(args: Tuple[PointPattern, Dict[str, Any]]) -> NumberVariance:
    pattern, kw = args
    return number_variance(
        pattern, radii=kw["radii"], n_windows=kw["n_windows"], seed=kw["seed"]
    )


def _run(
    worker: Callable[[Tuple[PointPattern, Dict[str, Any]]], Any],
    tasks: List[Tuple[PointPattern, Dict[str, Any]]],
    jobs: int,
) -> List[Any]:
    """Map `worker` over tasks serially or in a process pool, keeping order."""
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    if jobs == 1 or len(tasks) == 1:
        return [worker(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
        return list(pool.map(worker, tasks))


# ============================================================
# AVERAGING
# ============================================================


def _member_mean(
    values: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """NaN-aware mean, standard error, and member count along axis 0."""
    valid = np.isfinite(values)
    n = valid.sum(axis=0)
    filled = np.where(valid, values, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = filled.sum(axis=0) / n
        sq = np.where(valid, (values - mean) ** 2, 0.0).sum(axis=0)
        sem = np.sqrt(sq / (n - 1) / n)
    sem[n < 2] = np.nan
    return mean, sem, n.astype(np.int64)


def average_binned(results: Sequence[BinnedResult]) -> BinnedResult:
    """Average BinnedResults that share the same edges."""
    if not results:
        raise ValueError("nothing to average")
    if len(results) == 1:
        return results[0]
    edges = results[0].edges
    if any(
        r.edges.shape != edges.shape or not np.allclose(r.edges, edges) for r in results
    ):
        raise ValueError("all members must share the same bin edges")
    mean, sem, n = _member_mean(np.stack([r.mean for r in results]))
    x_mean, _, _ = _member_mean(np.stack([r.x_mean for r in results]))
    meta = dict(results[0].metadata)
    meta["n_members"] = len(results)
    meta["n_points"] = [r.metadata.get("n_points") for r in results]
    return BinnedResult(
        edges=edges, x_mean=x_mean, mean=mean, sem=sem, counts=n, metadata=meta
    )


def average_number_variance(results: Sequence[NumberVariance]) -> NumberVariance:
    """Average NumberVariance results that share the same radii."""
    if not results:
        raise ValueError("nothing to average")
    if len(results) == 1:
        return results[0]
    var, sem, _ = _member_mean(np.stack([r.variance for r in results]))
    mean_count, _, _ = _member_mean(np.stack([r.mean_count for r in results]))
    meta = dict(results[0].metadata)
    meta["n_members"] = len(results)
    meta["n_points"] = [r.metadata.get("n_points") for r in results]
    meta["seed"] = [r.metadata.get("seed") for r in results]
    return NumberVariance(
        radii=results[0].radii,
        variance=var,
        sem=sem,
        mean_count=mean_count,
        metadata=meta,
    )


# ============================================================
# ENSEMBLE
# ============================================================


class Ensemble:
    """
    Immutable, ordered collection of point patterns of one dimension.

    Parameters
    ----------
    patterns : iterable of PointPattern
        Members (at least one), all of the same spatial dimension.
    sources : sequence of str, optional
        Labels (e.g. file names) for the members, in the same order.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ens = Ensemble(PointPattern(rng.random((100, 2)), np.eye(2)) for _ in range(3))
    >>> g2 = ens.pair_correlation(dr=0.05)
    >>> g2.metadata["n_members"], g2.mean.shape
    (3, (10,))
    """

    __slots__ = ("_patterns", "_sources")

    def __init__(
        self, patterns: Iterable[PointPattern], sources: Optional[Sequence[str]] = None
    ) -> None:
        members = tuple(patterns)
        if not members:
            raise ValueError("an ensemble needs at least one pattern")
        if not all(isinstance(p, PointPattern) for p in members):
            raise TypeError("ensemble members must be PointPattern instances")
        dims = {p.dimension for p in members}
        if len(dims) != 1:
            raise ValueError(f"members have different dimensions: {sorted(dims)}")
        labels = (
            tuple(sources)
            if sources is not None
            else tuple(f"member_{i:04d}" for i in range(len(members)))
        )
        if len(labels) != len(members):
            raise ValueError("sources must have one label per pattern")
        self._patterns: Tuple[PointPattern, ...] = members
        self._sources: Tuple[str, ...] = labels

    # ----- construction -----

    @classmethod
    def from_files(
        cls,
        directory: Union[str, Path],
        glob: str = "*",
        fmt: str = "cco",
        **reader_kwargs: Any,
    ) -> "Ensemble":
        """
        Read every regular, non-hidden file matching `glob` in `directory`.

        Files are read in sorted (lexicographic) order, so member order is
        reproducible. For ``fmt="morsefire"`` matching *directories* are read.
        """
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; choose from {sorted(FORMATS)}")
        want_dirs = fmt == "morsefire"
        paths = sorted(
            p
            for p in root.glob(glob)
            if not p.name.startswith(".") and (p.is_dir() if want_dirs else p.is_file())
        )
        if not paths:
            raise FileNotFoundError(
                f"no {'directories' if want_dirs else 'files'} "
                f"matching {glob!r} in {root}"
            )
        reader = FORMATS[fmt]
        return cls((reader(p, **reader_kwargs) for p in paths), [p.name for p in paths])

    # ----- container protocol -----

    def __len__(self) -> int:
        return len(self._patterns)

    def __iter__(self) -> Iterator[PointPattern]:
        return iter(self._patterns)

    def __getitem__(self, index: int) -> PointPattern:
        return self._patterns[index]

    def __repr__(self) -> str:
        n = [p.n_points for p in self._patterns]
        return (
            f"Ensemble(members={len(self)}, dimension={self.dimension}, "
            f"N={min(n)}..{max(n)})"
        )

    @property
    def dimension(self) -> int:
        """Spatial dimension shared by all members."""
        return self._patterns[0].dimension

    @property
    def sources(self) -> Tuple[str, ...]:
        """Member labels (file names when read with `from_files`)."""
        return self._sources

    @property
    def mean_density(self) -> float:
        """Mean number density over members."""
        return float(np.mean([p.density for p in self._patterns]))

    # ----- statistics -----

    def structure_factor(
        self,
        k_max: Optional[float] = None,
        dk: Optional[float] = None,
        method: str = "nufft",
        jobs: int = 1,
        nthreads: int = 1,
    ) -> BinnedResult:
        """
        Ensemble-averaged, radially binned S(k).

        Defaults: `k_max` is the smallest member default (3 x 2 pi rho^(1/d));
        `dk` is the largest member reciprocal spacing, so every bin is populated
        for every member.
        """
        cut = min(default_k_max(p) for p in self) if k_max is None else float(k_max)
        width = max(default_dk(p.box) for p in self) if dk is None else float(dk)
        kw = {"k_max": cut, "dk": width, "method": method, "nthreads": nthreads}
        results = _run(_sf_worker, [(p, kw) for p in self], jobs)
        return average_binned(results)

    def pair_correlation(
        self, dr: Optional[float] = None, r_max: Optional[float] = None, jobs: int = 1
    ) -> BinnedResult:
        """Ensemble-averaged g2(r) on common bins (defaults from the members)."""
        width = (
            float(np.mean([default_dr(p) for p in self])) if dr is None else float(dr)
        )
        stop = min(default_r_max(p) for p in self) if r_max is None else float(r_max)
        kw = {"dr": width, "r_max": stop}
        return average_binned(_run(_g2_worker, [(p, kw) for p in self], jobs))

    def number_variance(
        self,
        radii: Optional[ArrayLike] = None,
        n_windows: int = 5000,
        seed: Optional[int] = None,
        jobs: int = 1,
    ) -> NumberVariance:
        """Ensemble-averaged sigma^2(R); member seeds are spawned from `seed`."""
        if radii is None:
            smallest = min(self, key=lambda p: p.box.inscribed_radius)
            r = default_radii(smallest)
        else:
            r = np.asarray(radii, dtype=np.float64)
        children = np.random.SeedSequence(seed).spawn(len(self))
        seeds = [int(c.generate_state(1)[0]) for c in children]
        tasks = [
            (p, {"radii": r, "n_windows": n_windows, "seed": s})
            for p, s in zip(self, seeds, strict=True)
        ]
        return average_number_variance(_run(_nv_worker, tasks, jobs))
