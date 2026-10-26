#!/usr/bin/env python3

"""
Integration tests: Ensemble, result files, and the command-line interface.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import subprocess
import sys
from pathlib import Path

# ----- third party -----
import numpy as np
import pytest

# ----- local modules -----
from npointstats import io
from npointstats.cli import main
from npointstats.ensemble import Ensemble, average_binned
from npointstats.pattern import PointPattern
from npointstats.results import load_table, save_result, save_structure_factor_raw
from npointstats.structure_factor import structure_factor

# ============================================================
# FIXTURES
# ============================================================


def write_ensemble(root: Path, members: int = 3, n: int = 400, d: int = 2) -> Path:
    """Write `members` Poisson patterns in CCO format with group-style names."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(members):
        io.write_cco(
            PointPattern(rng.random((n, d)), np.eye(d)), root / f"RSA__{i:04d}.txt"
        )
    return root


# ============================================================
# ENSEMBLE
# ============================================================


def test_from_files_sorted_and_filtered(tmp_path: Path) -> None:
    root = write_ensemble(tmp_path / "p", members=3)
    (root / ".hidden").write_text("junk")
    (root / "sub").mkdir()
    ens = Ensemble.from_files(root)
    assert ens.sources == ("RSA__0000.txt", "RSA__0001.txt", "RSA__0002.txt")
    assert len(ens) == 3
    assert ens.dimension == 2
    assert "members=3" in repr(ens)
    assert ens[0].n_points == 400
    with pytest.raises(FileNotFoundError, match="no files"):
        Ensemble.from_files(root, glob="*.dat")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        Ensemble.from_files(root / "missing")
    with pytest.raises(ValueError, match="unknown format"):
        Ensemble.from_files(root, fmt="xyz")


def test_ensemble_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Ensemble([])
    with pytest.raises(ValueError, match="dimensions"):
        Ensemble(
            [PointPattern([[0.1]], np.eye(1)), PointPattern([[0.1, 0.1]], np.eye(2))]
        )
    with pytest.raises(TypeError, match="PointPattern"):
        Ensemble([np.zeros((2, 2))])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="one label"):
        Ensemble([PointPattern([[0.1]], np.eye(1))], sources=["a", "b"])


def test_ensemble_statistics_poisson(tmp_path: Path) -> None:
    ens = Ensemble.from_files(write_ensemble(tmp_path / "p", members=4, n=1000))
    sf = ens.structure_factor()
    assert sf.metadata["n_members"] == 4
    assert sf.metadata["n_points"] == [1000] * 4
    assert np.all(sf.counts == 4)
    assert abs(np.average(sf.mean, weights=1 / sf.sem**2) - 1.0) < 0.05
    g2 = ens.pair_correlation()
    assert abs(np.nanmean(g2.mean[3:]) - 1.0) < 0.05
    nv = ens.number_variance(n_windows=500, seed=1)
    assert np.all(nv.variance > 0)
    assert nv.metadata["n_members"] == 4


def test_average_binned_rejects_mismatched_edges() -> None:
    p = PointPattern(np.random.default_rng(0).random((50, 2)), np.eye(2))
    a = structure_factor(p, k_max=60.0).radial_average(dk=2 * np.pi)
    b = structure_factor(p, k_max=60.0).radial_average(dk=np.pi)
    with pytest.raises(ValueError, match="same bin edges"):
        average_binned([a, b])
    with pytest.raises(ValueError, match="nothing"):
        average_binned([])


# ============================================================
# RESULT FILES
# ============================================================


def test_result_round_trip(tmp_path: Path) -> None:
    p = PointPattern(np.random.default_rng(1).random((200, 2)), np.eye(2))
    sk = structure_factor(p, k_max=80.0)
    path = save_result(sk.radial_average(), tmp_path / "sf.csv")
    data, columns, meta = load_table(path)
    assert columns[:5] == ["k_lo", "k_hi", "k", "k_mean", "S"]
    assert meta["statistic"] == "structure_factor"
    assert data.shape[1] == 7
    np.testing.assert_allclose(data[:, 4], sk.radial_average().to_array(True)[:, 4])
    raw = np.load(save_structure_factor_raw(sk, tmp_path / "raw.npz"))
    np.testing.assert_array_equal(raw["S"], sk.values)
    bad = tmp_path / "bad.csv"
    bad.write_text("1,2\n")
    with pytest.raises(ValueError, match="columns"):
        load_table(bad)


# ============================================================
# CLI
# ============================================================


def test_cli_end_to_end(tmp_path: Path) -> None:
    root = write_ensemble(tmp_path / "patterns", members=2, n=300)
    out = tmp_path / "out"
    status = main(
        [
            "-d",
            str(root),
            "-s",
            "sf",
            "g2",
            "nv",
            "--seed",
            "3",
            "--windows",
            "300",
            "-o",
            str(out),
            "-q",
            "--raw-sk",
        ]
    )
    assert status == 0
    for stat in ("sf", "g2", "nv"):
        data, _, meta = load_table(out / f"{stat}.csv")
        assert data.shape[0] > 0
        assert meta["sources"] == ["RSA__0000.txt", "RSA__0001.txt"]
    assert sorted(p.name for p in (out / "sk_raw").iterdir()) == [
        "RSA__0000.npz",
        "RSA__0001.npz",
    ]


def test_cli_simple_format_and_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "simple"
    root.mkdir()
    rng = np.random.default_rng(0)
    np.savetxt(root / "a.txt", 5.0 * rng.random((100, 2)))
    assert (
        main(
            [
                "-d",
                str(root),
                "-p",
                "simple",
                "--box",
                "5",
                "5",
                "-s",
                "g2",
                "-o",
                str(tmp_path / "o"),
                "-q",
            ]
        )
        == 0
    )
    assert main(["-d", str(root), "-p", "simple", "-s", "g2", "-q"]) == 1
    assert "needs --box" in capsys.readouterr().err
    assert main(["-d", str(root), "--box", "5", "-s", "g2", "-q"]) == 1
    assert main(["-d", str(tmp_path / "nope"), "-s", "g2", "-q"]) == 1


def test_cli_parallel_matches_serial(tmp_path: Path) -> None:
    root = write_ensemble(tmp_path / "p", members=3, n=300)
    for jobs in ("1", "2"):
        assert (
            main(
                [
                    "-d",
                    str(root),
                    "-s",
                    "sf",
                    "nv",
                    "--seed",
                    "9",
                    "--windows",
                    "200",
                    "-j",
                    jobs,
                    "-o",
                    str(tmp_path / f"o{jobs}"),
                    "-q",
                ]
            )
            == 0
        )
    for stat in ("sf", "nv"):
        a, _, _ = load_table(tmp_path / "o1" / f"{stat}.csv")
        b, _, _ = load_table(tmp_path / "o2" / f"{stat}.csv")
        np.testing.assert_array_equal(a, b)


def test_module_entry_point(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "npointstats", "--version"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    assert proc.stdout.startswith("npointstats ")
