#!/usr/bin/env python3

"""
Regression tests for defects found in code review of the refactor (D-047+).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import importlib
import sys

# ----- third party -----
import numpy as np

# ----- local modules -----
import npointstats

# ============================================================
# D-052: no submodule shadowed by a public attribute
# ============================================================


def test_regression_D052_no_submodule_shadowing() -> None:
    """D-052: every submodule is reachable as the attribute of the same name."""
    import pkgutil

    for info in pkgutil.iter_modules(npointstats.__path__):
        module = importlib.import_module(f"npointstats.{info.name}")
        assert sys.modules[f"npointstats.{info.name}"] is module
        attr = getattr(npointstats, info.name, module)
        assert attr is module, f"npointstats.{info.name} is shadowed by {attr!r}"
    # edge: the public functions are still exported at top level
    assert callable(npointstats.structure_factor)
    assert callable(npointstats.pair_correlation)
    assert callable(npointstats.number_variance)
    assert np.isfinite(npointstats.Box.cubic(1.0, 2).volume)
