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
from typing import Sequence

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ============================================================
# CONSTANTS
# ============================================================

# Relative tolerance for treating an off-diagonal lattice entry as zero.
ORTHORHOMBIC_RTOL = 1e-12

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

    __slots__ = ("_inverse", "_lattice")

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
    def face_widths(self) -> NDArray[np.float64]:
        """Distances between opposite faces of the cell, one per lattice vector."""
        return np.asarray(1.0 / np.linalg.norm(self._inverse, axis=0))

    @property
    def inscribed_radius(self) -> float:
        """
        Radius of the largest sphere that fits in the cell (half the smallest face
        width). Minimum-image distances are exact below this radius, and no pair is
        counted twice, so it bounds the range of g2(r) and number-variance windows.
        """
        return float(0.5 * np.min(self.face_widths))

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

    def minimum_image(self, dx: ArrayLike) -> NDArray[np.float64]:
        """
        Minimum-image form of displacement vectors `dx`.

        Rounds fractional components to the nearest integer. The result is the
        true shortest image whenever that image is shorter than
        `inscribed_radius`; otherwise it is some image at least as long, which
        is what pair statistics restricted to r < `inscribed_radius` need.
        """
        s = self.to_fractional(dx)
        s = s - np.rint(s)
        return np.asarray(s @ self._lattice)
