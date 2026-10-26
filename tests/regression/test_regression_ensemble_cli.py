#!/usr/bin/env python3

"""
Regression tests for ensemble, CLI, and packaging defects.

Covers D-003, D-005/D-031, D-006 .. D-010, D-021, D-023, D-024, D-029, D-030,
D-032.

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
from hypothesis import example, given, settings
from hypothesis import strategies as st

# ----- local modules -----
import npointstats
from npointstats import io
from npointstats.cli import main
from npointstats.ensemble import Ensemble
from npointstats.pattern import PointPattern
from npointstats.results import load_table
from npointstats.structure import structure_factor

# ============================================================
# HELPERS
# ============================================================

PACKAGE = Path(npointstats.__file__).resolve().parent


def poisson(n: int, length: float, seed: int) -> PointPattern:
    rng = np.random.default_rng(seed)
    return PointPattern(length * rng.random((n, 2)), length * np.eye(2))


def write_dir(root: Path, names: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        io.write_cco(poisson(200, 1.0, i), root / name)
    return root


# ============================================================
# ENSEMBLE (D-006 .. D-010)
# ============================================================


def test_regression_D006_no_shared_member_state() -> None:
    """D-006: ensembles never share members (immutable tuples, no defaults)."""
    a = Ensemble([poisson(10, 1.0, 0)])
    b = Ensemble([poisson(10, 1.0, 1)])
    assert len(a) == 1
    assert len(b) == 1
    assert a[0] is not b[0]
    # edge: there is no mutating API at all
    assert not hasattr(a, "_add_member")
    with pytest.raises(AttributeError):
        a.members = []  # type: ignore[attr-defined]


def test_regression_D007_parallel_path_runs_and_matches() -> None:
    """D-007: the parallel path works and equals the serial result."""
    ens = Ensemble([poisson(300, 1.0, s) for s in range(3)])
    serial = ens.structure_factor(jobs=1)
    parallel = ens.structure_factor(jobs=2)
    np.testing.assert_array_equal(serial.mean, parallel.mean)
    # edge: number variance seeds are per member, independent of jobs
    a = ens.number_variance(n_windows=200, seed=4, jobs=1)
    b = ens.number_variance(n_windows=200, seed=4, jobs=3)
    np.testing.assert_array_equal(a.variance, b.variance)


def test_regression_D008_g2_written_and_errors_raised(tmp_path: Path) -> None:
    """D-008: g2 is written; an unwritable output is an error, not silence."""
    root = write_dir(tmp_path / "p", ["a__0.txt", "a__1.txt"])
    assert main(["-d", str(root), "-s", "g2", "-o", str(tmp_path / "o"), "-q"]) == 0
    data, _, meta = load_table(tmp_path / "o" / "g2.csv")
    assert data.shape[0] > 0
    assert meta["statistic"] == "pair_correlation"
    # edge: output path is an existing *file* -> non-zero exit status
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    assert main(["-d", str(root), "-s", "g2", "-o", str(blocker), "-q"]) == 1


@given(st.floats(1.0, 3.0), st.integers(0, 1000))
@example(2.0, 0)
@settings(max_examples=15)
def test_regression_D009_average_is_binwise(length: float, seed: int) -> None:
    """D-009: members with different boxes are averaged bin by bin in |k|."""
    p1, p2 = poisson(100, 1.0, seed), poisson(int(100 * length**2), length, seed + 1)
    k_max, dk = 2 * np.pi * 6, 2 * np.pi
    ens = Ensemble([p1, p2]).structure_factor(k_max=k_max, dk=dk)
    r1 = structure_factor(p1, k_max=k_max).radial_average(dk=dk)
    r2 = structure_factor(p2, k_max=k_max).radial_average(dk=dk)
    np.testing.assert_allclose(ens.mean, 0.5 * (r1.mean + r2.mean), rtol=1e-12)
    np.testing.assert_array_equal(ens.edges, r1.edges)


def test_regression_D010_bin_width_is_a_parameter() -> None:
    """D-010: dk is user-controlled and the same for serial and parallel runs."""
    ens = Ensemble([poisson(200, 1.0, s) for s in range(2)])
    for dk in (np.pi, 2 * np.pi, 4 * np.pi):
        a = ens.structure_factor(dk=dk, jobs=1)
        b = ens.structure_factor(dk=dk, jobs=2)
        np.testing.assert_allclose(np.diff(a.edges), dk)
        np.testing.assert_array_equal(a.edges, b.edges)


# ============================================================
# CLI (D-003, D-023, D-024)
# ============================================================


def test_regression_D003_documented_invocation_runs(tmp_path: Path) -> None:
    """D-003: the documented command runs to completion (exit status 0)."""
    root = write_dir(tmp_path / "patterns", ["RSA__0000.txt"])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "npointstats",
            "-d",
            str(root),
            "-s",
            "sf",
            "-q",
            "-o",
            str(tmp_path / "o"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "o" / "sf.csv").is_file()


def test_regression_D023_default_glob_reads_group_file_names(tmp_path: Path) -> None:
    """D-023: by default every file is read, including 'NAME__0000.txt' names."""
    root = write_dir(tmp_path / "p", ["RSA-2D__0000.txt", "RSA-2D__0001.txt"])
    assert main(["-d", str(root), "-s", "g2", "-o", str(tmp_path / "o"), "-q"]) == 0
    _, _, meta = load_table(tmp_path / "o" / "g2.csv")
    assert meta["sources"] == ["RSA-2D__0000.txt", "RSA-2D__0001.txt"]
    # edge: a glob that matches nothing is a clear error
    assert main(["-d", str(root), "-n", "__*", "-s", "g2", "-q"]) == 1


def test_regression_D024_cli_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D-024: -s is required, --hdf5 is gone, and member order is sorted."""
    root = write_dir(tmp_path / "p", ["b.txt", "a.txt", "c.txt"])
    with pytest.raises(SystemExit) as info:
        main(["-d", str(root)])
    assert info.value.code == 2
    assert "--statistics" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["-d", str(root), "-s", "g2", "-f", "x.hdf5"])
    assert Ensemble.from_files(root).sources == ("a.txt", "b.txt", "c.txt")


# ============================================================
# PACKAGING (D-005, D-021, D-029 .. D-032)
# ============================================================


def test_regression_D031_python_shebangs() -> None:
    """D-031 (and D-005): every module starts with a Python shebang."""
    files = sorted(PACKAGE.rglob("*.py"))
    assert len(files) >= 10
    for path in files:
        assert path.read_text(encoding="utf-8").startswith(
            "#!/usr/bin/env python3\n"
        ), path


def test_regression_D029_errors_raise_instead_of_exiting(tmp_path: Path) -> None:
    """D-029: invalid input raises exceptions from the library, never SystemExit."""
    with pytest.raises(ValueError, match="unknown format"):
        io.read(tmp_path / "x", "nope")
    with pytest.raises(ValueError, match="method"):
        structure_factor(poisson(5, 1.0, 0), method="nope")
    bad = tmp_path / "bad.txt"
    bad.write_text("9\n")
    with pytest.raises(ValueError, match="lattice lines"):
        io.read_cco(bad)


def _python(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def test_regression_D030_importable_from_any_directory(tmp_path: Path) -> None:
    """D-030: the installed package imports without sys.path manipulation."""
    proc = _python("import sys, npointstats; print(len(sys.path))", tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert not any(
        "n_point_statistics" in line
        for line in PACKAGE.joinpath("__init__.py").read_text().splitlines()
        if "sys.path" in line
    )
    for path in PACKAGE.rglob("*.py"):
        assert "sys.path.append" not in path.read_text(encoding="utf-8"), path


def test_regression_D021_D032_no_jax_and_no_environment_side_effects(
    tmp_path: Path,
) -> None:
    """D-021/D-032: importing the package neither imports JAX nor edits os.environ."""
    code = (
        "import os, sys; before = dict(os.environ); import npointstats, "
        "npointstats.theory; print('jax' in sys.modules, dict(os.environ) == before)"
    )
    proc = _python(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["False", "True"]
