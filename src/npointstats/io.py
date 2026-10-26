#!/usr/bin/env python3

"""
Readers and writers for the point-pattern file formats used in the group.

Every reader returns a `PointPattern` with Cartesian positions. Supported
formats (key in `FORMATS`):

- ``cco``: CCOptimization text input. Line 1 is d; the next d lines are the
  lattice vectors (rows, padded to 4 columns); every remaining non-blank line
  is one point (first d columns used).
- ``dnv``: Donev event-driven MD output (``box::WriteConfiguration``): d;
  ``N n_species``; N per species; diameters; d*d lattice entries on one line;
  a ``T T T`` periodicity line; then one point per line.
- ``lstj``: Lubachevsky-Stillinger / Torquato-Jiao output: as ``dnv`` but with
  the lattice on d separate lines. Points start after the periodicity line
  (``T T T``); any line between the lattice and that line is skipped.

For ``dnv`` and ``lstj`` the number of points read must equal the N in the
header, so a layout mismatch raises instead of silently dropping a point.
- ``morsefire``: a directory holding ``boxSize.dat``, ``positions.dat`` and
  ``scalars.dat`` (FIRE minimizer output); orthorhombic box.
- ``simple``: whitespace- or comma-delimited columns of coordinates, ``#``
  comments allowed; the box must be supplied by the caller.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ----- local modules -----
from npointstats.box import Box
from npointstats.pattern import PointPattern

# ============================================================
# CONSTANTS
# ============================================================

PathLike = Union[str, Path]

# CCO pads every lattice vector and coordinate row to this many columns.
CCO_COLUMNS = 4

# ============================================================
# HELPERS
# ============================================================


def _nonblank_lines(path: PathLike) -> List[str]:
    """Return the non-blank lines of a text file, stripped of trailing whitespace."""
    with open(path, encoding="utf-8") as handle:
        lines = [line.rstrip() for line in handle]
    return [line for line in lines if line.strip()]


def _floats(line: str, count: int, where: str) -> List[float]:
    """Parse the first `count` numbers of a whitespace-separated line."""
    tokens = line.split()
    if len(tokens) < count:
        raise ValueError(f"{where}: expected at least {count} numbers, got {line!r}")
    try:
        return [float(tok) for tok in tokens[:count]]
    except ValueError as err:
        raise ValueError(f"{where}: could not parse numbers in {line!r}") from err


def _dimension(line: str, path: PathLike) -> int:
    """Parse the leading dimension field of a header line."""
    try:
        d = int(line.split()[0])
    except (ValueError, IndexError) as err:
        raise ValueError(f"{path}: first line must start with the dimension") from err
    if d < 1:
        raise ValueError(f"{path}: dimension must be >= 1, got {d}")
    return d


def _points(
    lines: List[str], start: int, d: int, path: PathLike
) -> NDArray[np.float64]:
    """Parse coordinate rows lines[start:] (first d columns each)."""
    body = lines[start:]
    if not body:
        raise ValueError(f"{path}: no point coordinates found")
    out = np.empty((len(body), d), dtype=np.float64)
    for i, line in enumerate(body):
        out[i] = _floats(line, d, f"{path}: line {start + i + 1}")
    return out


def _flag_line(lines: List[str], start: int, path: PathLike) -> int:
    """Index of the first periodicity-flag line (only T/F tokens) at or after start."""
    for i in range(start, len(lines)):
        tokens = lines[i].split()
        if tokens and all(tok.upper() in ("T", "F") for tok in tokens):
            return i
    raise ValueError(f"{path}: no periodicity line (e.g. 'T T T') found")


def _make_pattern(
    points: NDArray[np.float64], lattice: NDArray[np.float64], fractional: bool
) -> PointPattern:
    """Build a PointPattern, converting fractional coordinates if requested."""
    box = Box(lattice)
    positions = box.to_cartesian(points) if fractional else points
    return PointPattern(positions, box)


# ============================================================
# READERS
# ============================================================


def read_cco(path: PathLike) -> PointPattern:
    """
    Read a CCOptimization input file (Cartesian coordinates).

    Blank lines are ignored, so files with trailing newlines are accepted.
    """
    lines = _nonblank_lines(path)
    if not lines:
        raise ValueError(f"{path}: file is empty")
    d = _dimension(lines[0], path)
    if len(lines) < d + 2:
        raise ValueError(f"{path}: expected {d} lattice lines and at least one point")
    lattice = np.array(
        [_floats(lines[1 + i], d, f"{path}: lattice line {i + 1}") for i in range(d)]
    )
    return _make_pattern(_points(lines, d + 1, d, path), lattice, fractional=False)


def read_dnv(path: PathLike, fractional: bool = False) -> PointPattern:
    """
    Read a Donev event-driven MD configuration.

    Parameters
    ----------
    path : str or Path
        Input file.
    fractional : bool, default False
        Treat coordinates as fractional (relative to the lattice). The Donev
        engine writes a unit identity box, for which both readings coincide.

    Returns
    -------
    PointPattern
    """
    lines = _nonblank_lines(path)
    if len(lines) < 7:
        raise ValueError(f"{path}: too short for the Donev format")
    d = _dimension(lines[0], path)
    n_points = int(lines[1].split()[0])
    lattice = np.array(_floats(lines[4], d * d, f"{path}: lattice line")).reshape(d, d)
    points = _points(lines, _flag_line(lines, 5, path) + 1, d, path)
    if points.shape[0] != n_points:
        raise ValueError(f"{path}: header says N={n_points}, found {points.shape[0]}")
    return _make_pattern(points, lattice, fractional)


def read_lstj(path: PathLike, fractional: bool = False) -> PointPattern:
    """Read a Lubachevsky-Stillinger / Torquato-Jiao configuration (see `read_dnv`)."""
    lines = _nonblank_lines(path)
    if not lines:
        raise ValueError(f"{path}: file is empty")
    d = _dimension(lines[0], path)
    if len(lines) < 6 + d:
        raise ValueError(f"{path}: too short for the LSTJ format")
    n_points = int(lines[1].split()[0])
    lattice = np.array(
        [_floats(lines[4 + i], d, f"{path}: lattice line {i + 1}") for i in range(d)]
    )
    points = _points(lines, _flag_line(lines, 4 + d, path) + 1, d, path)
    if points.shape[0] != n_points:
        raise ValueError(f"{path}: header says N={n_points}, found {points.shape[0]}")
    return _make_pattern(points, lattice, fractional)


def read_morsefire(directory: PathLike) -> PointPattern:
    """
    Read FIRE-minimizer output from `directory`.

    Requires ``boxSize.dat`` (d side lengths, one or more per line),
    ``positions.dat`` (one point per line) and ``scalars.dat`` (tab-separated
    ``key value`` lines including ``nDim`` and ``numParticles``).
    """
    root = Path(directory)
    files = {
        name: root / name for name in ("boxSize.dat", "positions.dat", "scalars.dat")
    }
    missing = [name for name, file in files.items() if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"{root}: missing {', '.join(missing)}")
    scalars: Dict[str, float] = {}
    for line in _nonblank_lines(files["scalars.dat"]):
        key, _, value = line.partition("\t")
        scalars[key.strip()] = float(value)
    d = int(scalars["nDim"])
    n_points = int(scalars["numParticles"])
    lengths = np.loadtxt(files["boxSize.dat"], dtype=np.float64).ravel()
    if lengths.size != d:
        raise ValueError(f"{root}: boxSize.dat has {lengths.size} entries, nDim={d}")
    positions = np.loadtxt(files["positions.dat"], dtype=np.float64, ndmin=2)
    if positions.shape != (n_points, d):
        raise ValueError(
            f"{root}: positions.dat has shape {positions.shape}, "
            f"expected ({n_points}, {d})"
        )
    return PointPattern(positions, Box.from_lengths(lengths.tolist()))


def read_simple(
    path: PathLike,
    box: Union[Box, ArrayLike],
    columns: Optional[int] = None,
    delimiter: Optional[str] = None,
) -> PointPattern:
    """
    Read plain columns of coordinates; the periodic box must be given.

    Parameters
    ----------
    path : str or Path
        Text file, one point per line; ``#`` starts a comment.
    box : Box or array_like, shape (d, d)
        Periodic cell.
    columns : int, optional
        Use only the first `columns` columns (default: the box dimension).
    delimiter : str, optional
        Column separator (default: any whitespace; use ``","`` for CSV).

    Returns
    -------
    PointPattern
    """
    cell = box if isinstance(box, Box) else Box(box)
    ncol = cell.dimension if columns is None else columns
    if ncol != cell.dimension:
        raise ValueError(
            f"columns={ncol} does not match box dimension {cell.dimension}"
        )
    data = np.loadtxt(
        path, dtype=np.float64, delimiter=delimiter, comments="#", ndmin=2
    )
    if data.shape[1] < ncol:
        raise ValueError(
            f"{path}: expected at least {ncol} columns, got {data.shape[1]}"
        )
    return PointPattern(data[:, :ncol], cell)


# ============================================================
# WRITERS
# ============================================================


def _cco_row(values: NDArray[np.float64]) -> str:
    """Format one CCO row at full double precision, padded to 4 columns."""
    padded = list(values) + [0.0] * max(0, CCO_COLUMNS - len(values))
    return "\t".join(f"{v:.17g}" for v in padded) + "\t"


def write_cco(pattern: PointPattern, path: PathLike) -> Path:
    """
    Write `pattern` as a CCOptimization input file (Cartesian coordinates).

    Values are written with 17 significant digits, so `read_cco` recovers the
    positions and lattice bit for bit.
    """
    out = Path(path)
    rows = [str(pattern.dimension)]
    rows += [_cco_row(v) for v in pattern.box.lattice]
    rows += [_cco_row(p) for p in pattern.positions]
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out


# ============================================================
# REGISTRY
# ============================================================

FORMATS: Dict[str, Callable[..., PointPattern]] = {
    "cco": read_cco,
    "dnv": read_dnv,
    "lstj": read_lstj,
    "morsefire": read_morsefire,
    "simple": read_simple,
}


def read(path: PathLike, fmt: str = "cco", **kwargs: object) -> PointPattern:
    """
    Read a point pattern in any registered format.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "x.txt"
    >>> _ = write_cco(PointPattern([[0.1, 0.2]], Box.cubic(1.0, 2)), tmp)
    >>> read(tmp, "cco").positions
    array([[0.1, 0.2]])
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(FORMATS)}")
    return FORMATS[fmt](path, **kwargs)
