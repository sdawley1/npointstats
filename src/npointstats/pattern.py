#!/usr/bin/env python3

"""
Immutable container for a periodic point pattern.

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
from numpy.typing import ArrayLike, NDArray

# ----- local modules -----
from npointstats.box import Box

# ============================================================
# POINT PATTERN
# ============================================================


class PointPattern:
    """
    N points in a d-dimensional periodic box.

    Positions are Cartesian. They need not lie inside the primary cell; every
    statistic in this package is invariant under lattice translations of
    individual points. A read-only copy of the positions is stored, so the
    caller's array is never modified.

    Parameters
    ----------
    positions : array_like, shape (N, d)
        Cartesian coordinates. A 1D array of length N is accepted for d = 1.
    box : Box or array_like, shape (d, d)
        Periodic cell (a `Box`, or lattice vectors as rows).

    Examples
    --------
    >>> pattern = PointPattern([[0.25, 0.25], [0.75, 0.75]], Box.cubic(1.0, 2))
    >>> pattern.n_points, pattern.dimension, pattern.density
    (2, 2, 2.0)
    """

    __slots__ = ("_box", "_positions")

    def __init__(self, positions: ArrayLike, box: Box | ArrayLike) -> None:
        cell = box if isinstance(box, Box) else Box(box)
        arr = np.array(positions, dtype=np.float64, copy=True)
        if arr.ndim == 1 and cell.dimension == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or arr.shape[1] != cell.dimension:
            raise ValueError(
                f"positions must have shape (N, {cell.dimension}) to match the box, "
                f"got {arr.shape}"
            )
        if arr.shape[0] == 0:
            raise ValueError("a point pattern needs at least one point")
        if not np.all(np.isfinite(arr)):
            raise ValueError("positions must be finite")
        arr.setflags(write=False)
        self._positions: NDArray[np.float64] = arr
        self._box: Box = cell

    def __repr__(self) -> str:
        return (
            f"PointPattern(n_points={self.n_points}, dimension={self.dimension}, "
            f"density={self.density:.6g})"
        )

    def __len__(self) -> int:
        return self.n_points

    def __reduce__(self) -> Tuple[type, Tuple[NDArray[np.float64], Box]]:
        # rebuild through __init__ so unpickled copies are validated and read-only
        return (PointPattern, (np.array(self._positions), self._box))

    @property
    def positions(self) -> NDArray[np.float64]:
        """Read-only (N, d) Cartesian positions."""
        return self._positions

    @property
    def box(self) -> Box:
        """Periodic cell."""
        return self._box

    @property
    def n_points(self) -> int:
        """Number of points N."""
        return int(self._positions.shape[0])

    @property
    def dimension(self) -> int:
        """Spatial dimension d."""
        return self._box.dimension

    @property
    def density(self) -> float:
        """Number density rho = N / V."""
        return self.n_points / self._box.volume

    def wrapped(self) -> NDArray[np.float64]:
        """Positions mapped into the primary cell (new array)."""
        return self._box.wrap(self._positions)

    def fractional(self) -> NDArray[np.float64]:
        """Fractional coordinates wrapped into [0, 1)^d (new array)."""
        s = self._box.to_fractional(self._positions)
        s = s - np.floor(s)
        s[s >= 1.0] = 0.0
        return np.asarray(s)
