#!/usr/bin/env python3

"""
Command-line interface: ensemble statistics for a directory of patterns.

Examples
--------
Structure factor and g2 of every CCO file in ./patterns, 8 processes::

    npointstats -d patterns -s sf g2 --jobs 8 --out results/

Number variance of plain-text files in a 161.8 x 161.8 box::

    npointstats -d data -p simple --box 161.80216 161.80216 -s nv --seed 1

The console script calls `main`; a function entry point is required by
packaging, so this module deviates from the bare ``__main__`` house style.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# ----- third party -----
import numpy as np

# ----- local modules -----
from npointstats import __version__
from npointstats.box import Box
from npointstats.binning import BinnedResult
from npointstats.ensemble import Ensemble
from npointstats.io import FORMATS
from npointstats.variance import NumberVariance
from npointstats.results import save_result, save_structure_factor_raw
from npointstats.structure import structure_factor

# ============================================================
# CONSTANTS
# ============================================================

STATISTICS = {
    "sf": "structure factor S(k)",
    "g2": "pair correlation g2(r)",
    "nv": "number variance sigma^2(R)",
}

# ============================================================
# ARGUMENT PARSING
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the `npointstats` command."""
    parser = argparse.ArgumentParser(
        prog="npointstats",
        description="Ensemble n-point statistics of periodic point patterns.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # ----- input -----
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        required=True,
        help="directory holding the ensemble",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="*",
        help="glob for file names inside the directory (default: '*')",
    )
    parser.add_argument(
        "-p",
        "--format",
        default="cco",
        choices=sorted(FORMATS),
        help="file format (default: cco)",
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs="+",
        metavar="L",
        help="box side lengths (required for -p simple)",
    )
    parser.add_argument(
        "--fractional", action="store_true", help="dnv/lstj coordinates are fractional"
    )

    # ----- statistics -----
    parser.add_argument(
        "-s",
        "--statistics",
        nargs="+",
        required=True,
        choices=sorted(STATISTICS),
        help="statistics to compute",
    )
    parser.add_argument(
        "--k-max", type=float, help="S(k) cutoff (default 3 x 2 pi rho^(1/d))"
    )
    parser.add_argument("--dk", type=float, help="S(k) shell width (default: 2 pi / L)")
    parser.add_argument("--method", default="nufft", choices=["nufft", "direct"])
    parser.add_argument(
        "--nthreads", type=int, default=1, help="FINUFFT threads per member"
    )
    parser.add_argument(
        "--raw-sk",
        action="store_true",
        help="also write unbinned S(k) per member (.npz)",
    )
    parser.add_argument(
        "--dr", type=float, help="g2 bin width (default 0.05 rho^(-1/d))"
    )
    parser.add_argument(
        "--r-max", type=float, help="g2 range (default min(10 rho^(-1/d), L/2))"
    )
    parser.add_argument("--radii", type=float, nargs="+", help="number-variance radii")
    parser.add_argument("--windows", type=int, default=5000, help="windows per member")
    parser.add_argument("--seed", type=int, help="random seed for number variance")

    # ----- execution and output -----
    parser.add_argument("-j", "--jobs", type=int, default=1, help="parallel processes")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("."),
        help="output directory (created if missing)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="no progress messages"
    )
    return parser


# ============================================================
# DRIVER
# ============================================================


def _log(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def _reader_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Format-specific keyword arguments for the readers."""
    if args.format == "simple":
        if not args.box:
            raise ValueError("-p simple needs --box L1 [L2 ...]")
        return {"box": Box.from_lengths(args.box)}
    if args.box:
        raise ValueError("--box is only used with -p simple")
    if args.format in ("dnv", "lstj"):
        return {"fractional": args.fractional}
    return {}


def run(args: argparse.Namespace) -> List[Path]:
    """Compute the requested statistics; return the files written."""
    ensemble = Ensemble.from_files(
        args.directory, args.name, args.format, **_reader_kwargs(args)
    )
    _log(f"read {ensemble!r} from {args.directory}", args.quiet)
    args.out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for stat in dict.fromkeys(args.statistics):  # unique, in order
        start = time.perf_counter()
        result: Union[BinnedResult, NumberVariance]
        if stat == "sf":
            result = ensemble.structure_factor(
                k_max=args.k_max,
                dk=args.dk,
                method=args.method,
                jobs=args.jobs,
                nthreads=args.nthreads,
            )
        elif stat == "g2":
            result = ensemble.pair_correlation(
                dr=args.dr, r_max=args.r_max, jobs=args.jobs
            )
        else:
            radii = None if args.radii is None else np.asarray(args.radii)
            result = ensemble.number_variance(
                radii=radii, n_windows=args.windows, seed=args.seed, jobs=args.jobs
            )
        result.metadata["sources"] = list(ensemble.sources)
        written.append(save_result(result, args.out / f"{stat}.csv"))
        _log(
            f"{STATISTICS[stat]}: {time.perf_counter() - start:.2f} s -> {written[-1]}",
            args.quiet,
        )
    if args.raw_sk:
        raw = args.out / "sk_raw"
        raw.mkdir(exist_ok=True)
        for name, pattern in zip(ensemble.sources, ensemble, strict=True):
            sk = structure_factor(
                pattern, k_max=args.k_max, method=args.method, nthreads=args.nthreads
            )
            written.append(
                save_structure_factor_raw(sk, raw / f"{Path(name).stem}.npz")
            )
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point; returns the process exit status."""
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError) as err:
        print(f"npointstats: error: {err}", file=sys.stderr, flush=True)
        return 1
    return 0
