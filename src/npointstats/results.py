#!/usr/bin/env python3

"""
Saving and loading results as self-describing CSV and NPZ files.

CSV layout (readable with `load_table` or ``pandas.read_csv(path, comment="#")``)::

    # npointstats 0.1.0
    # statistic: "pair_correlation"
    # dr: 0.05
    # ...
    r_lo,r_hi,r,r_mean,g2,sem,count
    0,0.05,0.025,...

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

# ----- third party -----
import numpy as np
from numpy.typing import NDArray

# ----- local modules -----
from npointstats import __version__
from npointstats.binning import BinnedResult
from npointstats.variance import NumberVariance
from npointstats.structure import StructureFactor

# ============================================================
# CONSTANTS
# ============================================================

PathLike = Union[str, Path]

COLUMNS: Dict[str, List[str]] = {
    "structure_factor": ["k_lo", "k_hi", "k", "k_mean", "S", "sem", "count"],
    "pair_correlation": ["r_lo", "r_hi", "r", "r_mean", "g2", "sem", "count"],
    "number_variance": ["R", "variance", "sem", "mean_count"],
}

# ============================================================
# GENERIC TABLES
# ============================================================


def save_table(
    path: PathLike,
    table: NDArray[np.float64],
    columns: Sequence[str],
    metadata: Dict[str, object],
) -> Path:
    """
    Write a 2D table: ``#`` metadata lines, one header row, then CSV data.

    Values are written with 17 significant digits (lossless).
    """
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(columns):
        raise ValueError(
            f"table shape {arr.shape} does not match {len(columns)} columns"
        )
    if any("," in c or not c for c in columns):
        raise ValueError(f"column names must be non-empty and comma-free: {columns}")
    out = Path(path)
    lines = [f"# npointstats {__version__}"]
    for key, value in metadata.items():
        lines.append(f"# {key}: {json.dumps(value, default=str)}")
    lines.append(",".join(columns))
    lines += [",".join(f"{v:.17g}" for v in row) for row in arr]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_table(
    path: PathLike,
) -> Tuple[NDArray[np.float64], List[str], Dict[str, object]]:
    """Read a table written by `save_table`: (data, columns, metadata)."""
    meta: Dict[str, object] = {}
    columns: List[str] = []
    header_row = -1
    with open(path, encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if line.startswith("#"):
                text = line[1:].strip()
                if ":" in text:
                    key, value = text.split(":", 1)
                    meta[key.strip()] = json.loads(value)
                continue
            columns = line.strip().split(",")
            header_row = i
            break
    if header_row < 0 or not columns or not columns[0]:
        raise ValueError(f"{path}: missing column header row")
    data = np.loadtxt(path, delimiter=",", skiprows=header_row + 1, ndmin=2)
    if data.size == 0:
        data = np.empty((0, len(columns)))
    if data.shape[1] != len(columns):
        raise ValueError(f"{path}: {data.shape[1]} data columns, {len(columns)} names")
    return np.asarray(data), columns, meta


# ============================================================
# RESULT-SPECIFIC WRAPPERS
# ============================================================


def save_result(
    result: Union[BinnedResult, NumberVariance],
    path: PathLike,
    nonempty_only: bool = True,
) -> Path:
    """
    Save a binned statistic or a number variance as CSV.

    With `nonempty_only`, bins without data (NaN, e.g. S(k) shells holding no
    wave vector) are omitted; genuine zeros such as g2 in a hard core are kept.
    """
    if isinstance(result, NumberVariance):
        return save_table(
            path, result.to_array(), COLUMNS["number_variance"], result.metadata
        )
    stat = str(result.metadata.get("statistic", "binned"))
    columns = COLUMNS.get(stat, ["x_lo", "x_hi", "x", "x_mean", "mean", "sem", "count"])
    meta = dict(result.metadata)
    return save_table(path, result.to_array(nonempty_only=nonempty_only), columns, meta)


def save_structure_factor_raw(sk: StructureFactor, path: PathLike) -> Path:
    """Save unbinned S(k) (all wave vectors) to a compressed .npz file."""
    out = Path(path)
    np.savez_compressed(
        out,
        k_vectors=sk.k_vectors,
        k=sk.k,
        S=sk.values,
        k_max=sk.k_max,
        n_points=sk.n_points,
        lattice=sk.box.lattice,
        version=__version__,
    )
    return out
