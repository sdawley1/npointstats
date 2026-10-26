#!/usr/bin/env python3

"""
npointstats: n-point statistics of periodic point patterns.

Structure factor S(k), pair correlation function g2(r), and local number
variance sigma^2(R) of d-dimensional point patterns in periodic boxes of any
shape, plus liquid-state reference closures in `npointstats.theory`.

Sam Dawley
10/2026
"""

# ============================================================
# METADATA
# ============================================================

__version__ = "0.1.0"

# ============================================================
# PUBLIC API
# ============================================================

# ----- local modules -----
from npointstats.binning import BinnedResult, radial_average
from npointstats.box import Box
from npointstats.ensemble import Ensemble
from npointstats.io import read, write_cco
from npointstats.number_variance import (
    NumberVariance,
    number_variance,
    number_variance_from_sk,
)
from npointstats.pair_correlation import pair_correlation
from npointstats.pattern import PointPattern
from npointstats.results import load_table, save_result
from npointstats.structure_factor import StructureFactor, structure_factor

__all__ = [
    "BinnedResult",
    "Box",
    "Ensemble",
    "NumberVariance",
    "PointPattern",
    "StructureFactor",
    "__version__",
    "load_table",
    "number_variance",
    "number_variance_from_sk",
    "pair_correlation",
    "radial_average",
    "read",
    "save_result",
    "structure_factor",
    "write_cco",
]
