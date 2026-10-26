#!/usr/bin/env python3

"""
Turn a benchmark results JSON file into a Markdown table.

Usage: python benchmarks/summarize.py benchmarks/results-<job>.json

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
import statistics
import sys
from pathlib import Path

# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":

    rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print("| statistic | implementation | d | N | output size | time (s), median [min, max] | peak RSS (GB) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            cell = f"failed: {r['error']}"
            print(f"| {r['stat']} | {r['impl']} {r['kind']} | {r['d']} | {r['n']} | - | {cell} | - |")
            continue
        t = r["times"]
        cell = f"{statistics.median(t):.3g} [{min(t):.3g}, {max(t):.3g}]"
        print(
            f"| {r['stat']} | {r['impl']} {r['kind']} | {r['d']} | {r['n']:,} | "
            f"{r['size']:,} | {cell} | {r['peak_rss_gb']:.2f} |"
        )
