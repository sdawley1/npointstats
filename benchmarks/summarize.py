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
    header = [
        "statistic",
        "implementation",
        "d",
        "N",
        "output size",
        "time (s), median [min, max]",
        "peak RSS (GB)",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        lead = [str(r["stat"]), f"{r['impl']} {r['kind']}", str(r["d"]), f"{r['n']:,}"]
        if "error" in r:
            cells = [*lead, "-", f"failed: {r['error']}", "-"]
        else:
            t = r["times"]
            span = f"{statistics.median(t):.3g} [{min(t):.3g}, {max(t):.3g}]"
            cells = [*lead, f"{r['size']:,}", span, f"{r['peak_rss_gb']:.2f}"]
        print("| " + " | ".join(cells) + " |")
