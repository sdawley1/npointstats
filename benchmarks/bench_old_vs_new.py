#!/usr/bin/env python3

"""
Benchmark the original (pre-refactor) code against npointstats.

Each case runs in its own subprocess so its peak resident memory is measured
in isolation. Times are the median and min/max of `--repeat` calls after one
untimed warm-up call (which also absorbs JAX JIT compilation in the legacy
code). Legacy code is extracted from git (`--legacy-rev`) into a temporary
directory and run with `--legacy-python` (an environment with jax + finufft).

Run on a compute node, not a login node, e.g.::

    sbatch benchmarks/run_benchmarks.slurm

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================
# CASE TEMPLATES
# ============================================================

HEADER = """
import json, resource, sys, time
import numpy as np
x = np.random.default_rng(0).random(({n}, {d}))
lv = np.eye({d})
"""

FOOTER = """
fn()
times = []
for _ in range({repeat}):
    t = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(json.dumps({{"times": times, "peak_rss_gb": rss, "size": size(out)}}))
"""

LEGACY = {
    "g2": """
sys.path.insert(0, {src!r})
from src.cpu_analysis.c_npt_correlation import sample_xd_g2
a = {n} ** (-1 / {d})
fn = lambda: sample_xd_g2(x, lv, delta_r=0.05 * a)
size = lambda out: int(out.shape[0])
""",
    "sk_collective": """
sys.path.insert(0, {src!r})
from src.cpu_analysis.c_structure_factor import sample_xd_sf_collective
fn = lambda: sample_xd_sf_collective(x, lv)
size = lambda out: int(out.shape[0])
""",
    "sk_finufft": """
sys.path.insert(0, {src!r})
from src.cpu_analysis.c_structure_factor import sample_xd_sf
fn = lambda: sample_xd_sf(x.copy(), lv)
size = lambda out: int(out.shape[0])
""",
}

NEW = {
    "g2": """
from npointstats import PointPattern, pair_correlation
p = PointPattern(x, lv); a = {n} ** (-1 / {d})
fn = lambda: pair_correlation(p, dr=0.05 * a, r_max=0.5)
size = lambda out: int(out.mean.size)
""",
    "g2_default": """
from npointstats import PointPattern, pair_correlation
p = PointPattern(x, lv)
fn = lambda: pair_correlation(p)
size = lambda out: int(out.mean.size)
""",
    "sk_nufft": """
from npointstats import PointPattern, structure_factor
p = PointPattern(x, lv)
fn = lambda: structure_factor(p, k_max={k_max})
size = lambda out: int(out.values.size)
""",
}

# (name, implementation, kind, n, d, extra) - k_max chosen to match the
# legacy wave-vector extent: 2D collective kn=250, 2D finufft kn=500,
# 3D collective kn=15 (legacy samples one quadrant/octant only).
CASES = (
    [("g2", "legacy", "g2", n, 2, {}) for n in (1000, 10000, 30000)]
    + [("g2", "new", "g2", n, 2, {}) for n in (1000, 10000, 30000)]
    + [("g2", "new", "g2_default", n, 2, {}) for n in (10000, 100000)]
    + [
        ("g2", "legacy", "g2", 10000, 3, {}),
        ("g2", "new", "g2", 10000, 3, {}),
    ]
    + [("sk", "legacy", "sk_collective", n, 2, {}) for n in (1000, 10000)]
    + [("sk", "legacy", "sk_finufft", n, 2, {}) for n in (1000, 10000, 100000)]
    + [
        ("sk", "new", "sk_nufft", n, 2, {"k_max": "2 * np.pi * 249"})
        for n in (1000, 10000, 100000)
    ]
    + [
        ("sk", "legacy", "sk_collective", 10000, 3, {}),
        ("sk", "new", "sk_nufft", 10000, 3, {"k_max": "2 * np.pi * 14"}),
    ]
)

# ============================================================
# HELPERS
# ============================================================


def extract_legacy(rev: str, repo: Path, dest: Path) -> Path:
    """Extract `rev` of `repo` into `dest` with git archive; return the root."""
    blob = subprocess.run(
        ["git", "-C", str(repo), "archive", rev], check=True, capture_output=True
    ).stdout
    with tarfile.open(fileobj=BytesIO(blob)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def run_case(python: str, code: str, timeout: float) -> Optional[Dict[str, object]]:
    """Run one benchmark subprocess; None on failure or timeout."""
    try:
        proc = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout:.0f} s"}
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["?"])[-1]
        return {"error": f"exit {proc.returncode}: {tail[:200]}"}
    return dict(json.loads(proc.stdout.strip().splitlines()[-1]))


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])

    # ----- interpreters -----
    parser.add_argument("--legacy-python", required=True)
    parser.add_argument("--new-python", default=sys.executable)

    # ----- options -----
    parser.add_argument("--legacy-rev", default="363ae8f")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results.json"))

    args = parser.parse_args()

    # =========================================================
    # RUN
    # =========================================================
    repo = Path(__file__).resolve().parents[1]
    results: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        src = str(extract_legacy(args.legacy_rev, repo, Path(tmp)))
        for stat, impl, kind, n, d, extra in CASES:
            fmt = {"n": n, "d": d, "src": src, "repeat": args.repeat, **extra}
            body = (LEGACY if impl == "legacy" else NEW)[kind]
            code = (HEADER + body + FOOTER).format(**fmt)
            python = args.legacy_python if impl == "legacy" else args.new_python
            res = run_case(python, code, args.timeout)
            row = {
                "stat": stat,
                "impl": impl,
                "kind": kind,
                "n": n,
                "d": d,
                **(res or {}),
            }
            results.append(row)
            print(json.dumps(row), flush=True)

    args.out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
