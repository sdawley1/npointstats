#!/usr/bin/env python3

"""
Quickstart: S(k), g2(r) and sigma^2(R) of a Poisson pattern and a crystal.

Run with ``python examples/quickstart.py``; it writes CSV files to a temporary
directory and prints a short summary. Executed by CI, so it must stay fast.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import tempfile
from pathlib import Path

# ----- third party -----
import numpy as np

# ----- local modules -----
import npointstats as nps

# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    # =========================================================
    # BUILD PATTERNS
    # =========================================================
    rng = np.random.default_rng(0)
    box = nps.Box.cubic(1.0, 2)
    poisson = nps.PointPattern(rng.random((2000, 2)), box)

    m = 32
    grid = np.stack(np.meshgrid(np.arange(m), np.arange(m)), -1).reshape(-1, 2) / m
    crystal = nps.PointPattern(grid + 0.01 / m * rng.standard_normal(grid.shape), box)

    # =========================================================
    # SINGLE-PATTERN STATISTICS
    # =========================================================
    sk = nps.structure_factor(poisson)  # every k with 0 < |k| <= 6 pi rho^(1/2)
    binned = sk.radial_average()
    g2 = nps.pair_correlation(poisson)
    nv = nps.number_variance(poisson, n_windows=2000, seed=1)
    nv_sk = nps.number_variance_from_sk(nps.structure_factor(crystal), nv.radii[:5])

    print(f"{poisson!r}", flush=True)
    print(
        f"  <S(k)> over {sk.values.size} wave vectors = {sk.values.mean():.3f}",
        flush=True,
    )
    print(f"  g2 at r ~ {g2.x_mean[-1]:.3f}: {g2.mean[-1]:.3f}", flush=True)
    print(
        f"  sigma^2(R) / <N(R)> at R = {nv.radii[-1]:.3f}: "
        f"{nv.variance[-1] / nv.mean_count[-1]:.3f}",
        flush=True,
    )
    print(f"crystal sigma^2(R) from S(k): {np.round(nv_sk.variance, 4)}", flush=True)

    # =========================================================
    # ENSEMBLE AND OUTPUT FILES
    # =========================================================
    ensemble = nps.Ensemble(
        nps.PointPattern(rng.random((1000, 2)), box) for _ in range(4)
    )
    out = Path(tempfile.mkdtemp(prefix="npointstats_"))
    nps.save_result(ensemble.structure_factor(), out / "sf.csv")
    nps.save_result(ensemble.pair_correlation(), out / "g2.csv")
    data, columns, meta = nps.load_table(out / "sf.csv")
    print(
        f"ensemble S(k): {data.shape[0]} bins, columns {columns}, "
        f"{meta['n_members']} members -> {out}",
        flush=True,
    )
