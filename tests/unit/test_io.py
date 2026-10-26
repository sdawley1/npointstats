#!/usr/bin/env python3

"""
Unit tests for npointstats.io readers and writers.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path

# ----- third party -----
import numpy as np
import pytest

# ----- local modules -----
from npointstats import io
from npointstats.box import Box
from npointstats.pattern import PointPattern

# ============================================================
# CONSTANTS
# ============================================================

DATA = Path(__file__).resolve().parents[1] / "data"

DNV_TEXT = """2
3 1
3
0.1
2 0 0 2
T T
0.1 0.2
1.0 1.5
1.9 0.3
"""

LSTJ_TEXT = """2 hs
3 1
3
0.1
2 0
0 2
T T
0.1 0.2
1.0 1.5
1.9 0.3
"""

# ============================================================
# CCO
# ============================================================


def test_read_real_cco_files() -> None:
    chain = io.read_cco(DATA / "cco_chain_1d_N100.out")
    assert chain.n_points == 100
    assert chain.dimension == 1
    np.testing.assert_allclose(np.diff(chain.positions[:, 0]), 0.01, atol=1e-9)
    csl = io.read_cco(DATA / "cco_csl_3d_N864.out")
    assert (csl.n_points, csl.dimension) == (864, 3)
    np.testing.assert_allclose(csl.box.lattice, np.eye(3))


def test_cco_write_format(tmp_path: Path) -> None:
    pattern = PointPattern([[0.5, 0.25]], Box.cubic(1.0, 2))
    text = io.write_cco(pattern, tmp_path / "a.txt").read_text().splitlines()
    assert text[0] == "2"
    assert text[1].split() == ["1", "0", "0", "0"]
    assert text[3].split() == ["0.5", "0.25", "0", "0"]


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("", "empty"),
        ("x\n1 0\n0 1\n0 0\n", "dimension"),
        ("2\n1 0\n0 1\n", "lattice lines"),
        ("2\n1 0\n0 1\n0.5\n", "at least 2 numbers"),
        ("2\n1 0\n0 1\n0.5 abc\n", "could not parse"),
    ],
)
def test_cco_errors(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        io.read_cco(path)


# ============================================================
# DONEV AND LSTJ
# ============================================================


@pytest.mark.parametrize(
    ("reader", "text"), [(io.read_dnv, DNV_TEXT), (io.read_lstj, LSTJ_TEXT)]
)
def test_read_dnv_and_lstj(tmp_path: Path, reader, text: str) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "conf.dat"
    path.write_text(text)
    pattern = reader(path)
    np.testing.assert_allclose(pattern.box.lattice, 2 * np.eye(2))
    np.testing.assert_allclose(pattern.positions[1], [1.0, 1.5])
    frac = reader(path, fractional=True)
    np.testing.assert_allclose(frac.positions[1], [2.0, 3.0])


def test_dnv_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "conf.dat"
    path.write_text(DNV_TEXT.replace("3 1\n", "4 1\n", 1))
    with pytest.raises(ValueError, match="header says N=4"):
        io.read_dnv(path)


def test_dnv_missing_flags(tmp_path: Path) -> None:
    path = tmp_path / "conf.dat"
    path.write_text(DNV_TEXT.replace("T T\n", "0 0\n"))
    with pytest.raises(ValueError, match="periodicity"):
        io.read_dnv(path)


def test_dnv_too_short(tmp_path: Path) -> None:
    path = tmp_path / "conf.dat"
    path.write_text("2\n1 1\n")
    with pytest.raises(ValueError, match="too short"):
        io.read_dnv(path)


# ============================================================
# MORSEFIRE AND SIMPLE
# ============================================================


def _write_morse(root: Path, d: int, n: int) -> None:
    root.mkdir()
    (root / "boxSize.dat").write_text("\n".join(["2.0"] * d) + "\n")
    rng = np.random.default_rng(0)
    np.savetxt(root / "positions.dat", 2.0 * rng.random((n, d)))
    (root / "scalars.dat").write_text(f"nDim\t{d}\nnumParticles\t{n}\n")
    (root / "radii.dat").write_text("0.1\n" * n)


def test_read_morsefire_2d(tmp_path: Path) -> None:
    _write_morse(tmp_path / "run", 2, 5)
    pattern = io.read_morsefire(tmp_path / "run")
    assert pattern.n_points == 5
    np.testing.assert_allclose(pattern.box.lattice, 2 * np.eye(2))


def test_read_morsefire_shape_mismatch(tmp_path: Path) -> None:
    _write_morse(tmp_path / "run", 2, 5)
    (tmp_path / "run" / "scalars.dat").write_text("nDim\t2\nnumParticles\t6\n")
    with pytest.raises(ValueError, match=r"positions\.dat"):
        io.read_morsefire(tmp_path / "run")


def test_read_simple_csv_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "pts.csv"
    path.write_text("# header\n0.1,0.2,9\n0.3,0.4,9\n")
    pattern = io.read_simple(path, Box.cubic(1.0, 2), delimiter=",")
    np.testing.assert_allclose(pattern.positions, [[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="columns"):
        io.read_simple(path, np.eye(2), columns=(0, 1, 2), delimiter=",")
    with pytest.raises(ValueError, match="at least 4 columns"):
        io.read_simple(path, np.eye(4), delimiter=",")


def test_read_dispatch(tmp_path: Path) -> None:
    path = io.write_cco(PointPattern([[0.1]], np.eye(1)), tmp_path / "c.txt")
    assert io.read(path, "cco").n_points == 1
    with pytest.raises(ValueError, match="unknown format"):
        io.read(path, "xyz")
