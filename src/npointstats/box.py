#!/usr/bin/env python3

"""
Periodic simulation cells of arbitrary shape.

A `Box` is defined by its lattice vectors, stored as the *rows* of a (d, d)
matrix L. A point with fractional coordinates s has Cartesian position
x = s @ L. Every geometric operation (wrapping, minimum image, reciprocal
lattice) is done in fractional coordinates, so it is correct for triclinic
(sheared, rhombic, ...) cells and not only for axis-aligned ones.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import itertools
from typing import Sequence, Tuple

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ============================================================
# CONSTANTS
# ============================================================

# Relative tolerance for treating an off-diagonal lattice entry as zero.
ORTHORHOMBIC_RTOL = 1e-12

# Lovasz parameter of the LLL reduction.
LLL_DELTA = 0.99

# Hard bound on LLL swap iterations (never reached for well-posed lattices).
LLL_MAX_ITER = 10_000

# ============================================================
# LATTICE REDUCTION
# ============================================================


def _gram_schmidt(
    b: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gram-Schmidt vectors b* (rows) and coefficients mu of the rows of b."""
    d = b.shape[0]
    bstar = np.zeros_like(b)
    mu = np.zeros((d, d))
    for i in range(d):
        bstar[i] = b[i]
        for j in range(i):
            mu[i, j] = b[i] @ bstar[j] / (bstar[j] @ bstar[j])
            bstar[i] = bstar[i] - mu[i, j] * bstar[j]
    return bstar, mu


def lll_reduce(lattice: ArrayLike) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    LLL-reduce the rows of `lattice` (Lenstra, Lenstra & Lovasz 1982), then
    apply a greedy shortening pass (Minkowski-reduced for d <= 4).

    Returns
    -------
    reduced : ndarray, shape (d, d)
        Reduced basis of the same lattice, ``reduced = transform @ lattice``.
    transform : ndarray of int, shape (d, d)
        Unimodular integer matrix (determinant +-1).

    Examples
    --------
    >>> reduced, u = lll_reduce([[30.0, 0.0], [27.0, 3.0]])
    >>> np.abs(reduced).tolist(), round(abs(float(np.linalg.det(u))))
    ([[3.0, 3.0], [15.0, 15.0]], 1)
    """
    b = np.array(lattice, dtype=np.float64)
    d = b.shape[0]
    u = np.eye(d, dtype=np.int64)
    k, iterations = 1, 0
    while k < d:
        iterations += 1
        if iterations > LLL_MAX_ITER:
            raise RuntimeError("LLL reduction did not converge")
        for j in range(k - 1, -1, -1):
            _, mu = _gram_schmidt(b)
            q = int(np.rint(mu[k, j]))
            if q != 0:
                b[k] -= q * b[j]
                u[k] -= q * u[j]
        bstar, mu = _gram_schmidt(b)
        lhs = bstar[k] @ bstar[k]
        rhs = (LLL_DELTA - mu[k, k - 1] ** 2) * (bstar[k - 1] @ bstar[k - 1])
        if lhs >= rhs:
            k += 1
        else:
            b[[k - 1, k]] = b[[k, k - 1]]
            u[[k - 1, k]] = u[[k, k - 1]]
            k = max(k - 1, 1)
    u = _greedy_improve(u, np.asarray(lattice, dtype=np.float64))
    # recompute from the integer transform so the basis is exact up to rounding
    reduced = u.astype(np.float64) @ np.asarray(lattice, dtype=np.float64)
    return reduced, u


def _greedy_improve(
    u: NDArray[np.int64], lattice: NDArray[np.float64]
) -> NDArray[np.int64]:
    """
    Greedy pass after LLL: sort rows by length and shorten each row by adding
    small integer combinations (coefficients in -2..2) of the shorter rows.

    For d <= 4 greedy reduction yields a Minkowski-reduced basis (Nguyen and
    Stehle, ACM Trans. Algorithms 5, 46 (2009)), which makes derived quantities
    such as the inscribed radius independent of the input basis.
    """
    d = u.shape[0]
    u = u.copy()
    coeffs = [
        np.array(c) for c in itertools.product(range(-2, 3), repeat=max(d - 1, 1))
    ]
    for _ in range(LLL_MAX_ITER):
        b = u.astype(np.float64) @ lattice
        order = np.argsort(np.einsum("ij,ij->i", b, b), kind="stable")
        u, b = u[order], b[order]
        changed = False
        for i in range(1, d):
            best = float(b[i] @ b[i])
            for c in coeffs:
                cand = b[i] + c[:i] @ b[:i]
                norm = float(cand @ cand)
                if norm < best * (1 - 1e-12):
                    best, b[i] = norm, cand
                    u[i] = u[i] + c[:i] @ u[:i]
                    changed = True
        if not changed:
            return u
    raise RuntimeError("greedy lattice reduction did not converge")


# ============================================================
# BOX
# ============================================================


class Box:
    """
    Periodic cell spanned by d lattice vectors (rows of `lattice`).

    Parameters
    ----------
    lattice : array_like, shape (d, d)
        Lattice vectors as rows. Must be finite and non-singular. A read-only
        copy is stored, so later changes to the input do not affect the box.

    Examples
    --------
    >>> box = Box([[1.0, 0.0], [0.5, 3**0.5 / 2]])
    >>> round(box.volume, 6)
    0.866025
    >>> box.minimum_image([0.5, 3**0.5 / 2]).round(12) + 0.0
    array([0., 0.])
    """

    __slots__ = ("_inverse", "_lattice", "_reduced", "_reduced_inverse")

    def __init__(self, lattice: ArrayLike) -> None:
        arr = np.array(lattice, dtype=np.float64, copy=True)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
            raise ValueError(
                f"lattice must be a non-empty (d, d) array, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("lattice must be finite")
        det = np.linalg.det(arr)
        scale = float(np.prod(np.linalg.norm(arr, axis=1)))
        if scale == 0.0 or abs(det) <= 1e-12 * scale:
            raise ValueError(
                "lattice vectors are linearly dependent (singular lattice)"
            )
        arr.setflags(write=False)
        inverse = np.linalg.inv(arr)
        inverse.setflags(write=False)
        self._lattice: NDArray[np.float64] = arr
        self._inverse: NDArray[np.float64] = inverse
        # geometry (minimum image, radius limits, k grids) uses a reduced basis
        # of the same lattice, so results do not depend on the basis supplied
        reduced = arr.copy() if self.is_orthorhombic else lll_reduce(arr)[0]
        reduced.setflags(write=False)
        reduced_inverse = np.linalg.inv(reduced)
        reduced_inverse.setflags(write=False)
        self._reduced: NDArray[np.float64] = reduced
        self._reduced_inverse: NDArray[np.float64] = reduced_inverse

    # ----- constructors -----

    @classmethod
    def cubic(cls, length: float, dimension: int) -> "Box":
        """Hypercubic box of side `length` in `dimension` dimensions."""
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        return cls(float(length) * np.eye(dimension))

    @classmethod
    def from_lengths(cls, lengths: Sequence[float]) -> "Box":
        """Axis-aligned orthorhombic box with the given side lengths."""
        return cls(np.diag(np.asarray(lengths, dtype=np.float64)))

    # ----- dunder methods -----

    def __repr__(self) -> str:
        return f"Box({self._lattice.tolist()!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return NotImplemented
        return bool(np.array_equal(self._lattice, other._lattice))

    def __hash__(self) -> int:
        return hash(self._lattice.tobytes())

    def __reduce__(self) -> Tuple[type, Tuple[NDArray[np.float64]]]:
        # rebuild through __init__ so unpickled copies are validated and read-only
        return (Box, (np.array(self._lattice),))

    # ----- properties -----

    @property
    def lattice(self) -> NDArray[np.float64]:
        """Read-only (d, d) array of lattice vectors (rows)."""
        return self._lattice

    @property
    def dimension(self) -> int:
        """Spatial dimension d."""
        return int(self._lattice.shape[0])

    @property
    def volume(self) -> float:
        """Cell volume |det L|."""
        return float(abs(np.linalg.det(self._lattice)))

    @property
    def reciprocal(self) -> NDArray[np.float64]:
        """Reciprocal lattice vectors b_j (rows) with a_i . b_j = 2 pi delta_ij."""
        return np.asarray(2.0 * np.pi * self._inverse.T)

    @property
    def reduced_lattice(self) -> NDArray[np.float64]:
        """LLL-reduced basis (rows) of the same lattice; `lattice` if diagonal."""
        return self._reduced

    @property
    def reduced_reciprocal(self) -> NDArray[np.float64]:
        """Reciprocal basis dual to `reduced_lattice`."""
        return np.asarray(2.0 * np.pi * self._reduced_inverse.T)

    @property
    def face_widths(self) -> NDArray[np.float64]:
        """Distances between opposite faces of the reduced cell."""
        return np.asarray(1.0 / np.linalg.norm(self._reduced_inverse, axis=0))

    @property
    def inscribed_radius(self) -> float:
        """
        Radius of the largest sphere inside the reduced cell (half its smallest
        face width). Minimum-image distances are exact below this radius and no
        pair is counted twice, so it bounds the range of g2(r) and of
        number-variance windows. It depends only on the lattice, not on the
        basis used to describe it.
        """
        return float(0.5 * np.min(self.face_widths))

    @property
    def shortest_reciprocal_norm(self) -> float:
        """Length of the shortest non-zero reciprocal lattice vector."""
        basis = lll_reduce(self.reciprocal)[0]
        d = self.dimension
        grid = np.stack(
            [g.ravel() for g in np.meshgrid(*[np.arange(-2, 3)] * d, indexing="ij")],
            axis=1,
        )
        norms = np.linalg.norm(grid[np.any(grid != 0, axis=1)] @ basis, axis=1)
        return float(np.min(norms))

    @property
    def is_orthorhombic(self) -> bool:
        """True if the lattice is diagonal (axis-aligned rectangular cell)."""
        off = self._lattice - np.diag(np.diag(self._lattice))
        scale = float(np.max(np.abs(self._lattice)))
        return bool(np.all(np.abs(off) <= ORTHORHOMBIC_RTOL * scale)) and bool(
            np.all(np.diag(self._lattice) > 0)
        )

    # ----- coordinate transforms -----

    def _check_points(self, x: ArrayLike) -> NDArray[np.float64]:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0 or arr.shape[-1] != self.dimension:
            raise ValueError(
                f"points must have trailing dimension {self.dimension}, got {arr.shape}"
            )
        return arr

    def to_fractional(self, x: ArrayLike) -> NDArray[np.float64]:
        """Fractional coordinates s of Cartesian points x (x = s @ L)."""
        return np.asarray(self._check_points(x) @ self._inverse)

    def to_cartesian(self, s: ArrayLike) -> NDArray[np.float64]:
        """Cartesian coordinates of fractional coordinates s."""
        return np.asarray(self._check_points(s) @ self._lattice)

    def wrap(self, x: ArrayLike) -> NDArray[np.float64]:
        """Map Cartesian points into the primary cell (fractional coords in [0, 1))."""
        s = self.to_fractional(x)
        s = s - np.floor(s)
        # floor can return exactly 1.0 after rounding for tiny negative inputs
        s[s >= 1.0] = 0.0
        return np.asarray(s @ self._lattice)

    def to_reduced_fractional(self, x: ArrayLike) -> NDArray[np.float64]:
        """Fractional coordinates of Cartesian points in the reduced basis."""
        return np.asarray(self._check_points(x) @ self._reduced_inverse)

    def minimum_image(self, dx: ArrayLike) -> NDArray[np.float64]:
        """
        Minimum-image form of displacement vectors `dx`.

        Rounds fractional components in the reduced basis to the nearest
        integer. The result is the true shortest image whenever that image is
        shorter than `inscribed_radius`; otherwise it is some image at least as
        long, which is what statistics restricted to r < `inscribed_radius` need.
        """
        s = self.to_reduced_fractional(dx)
        s = s - np.rint(s)
        return np.asarray(s @ self._reduced)
