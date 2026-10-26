#!/usr/bin/env python3

"""
Regression and round-trip property tests for file I/O defects.

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
import pytest
from hypothesis import example, given
from npst_strategies import patterns

# ----- local modules -----
from npointstats import io
from npointstats.box import Box
from npointstats.pattern import PointPattern

# ============================================================
# D-025: simple reader must use the caller's box
# ============================================================


def test_regression_D025_simple_reader_uses_given_box(tmp_path: Path) -> None:
    """D-025: a 3D 'simple' file gets the supplied 3D box, not a 2D identity."""
    path = tmp_path / "pts.txt"
    path.write_text("0.1 0.2 0.3\n1.0 2.0 3.0\n")
    box = Box.from_lengths([2.0, 3.0, 4.0])
    pattern = io.read_simple(path, box)
    assert pattern.box == box
    assert pattern.positions.shape == (2, 3)
    # edge: box size unrelated to 161.80216 (old hard-coded chiral-active box)
    two_d = io.read_simple(path, Box.cubic(5.0, 2))
    assert two_d.box.volume == pytest.approx(25.0)


# ============================================================
# D-026: blank lines in CCO files
# ============================================================


def test_regression_D026_cco_trailing_blank_lines(tmp_path: Path) -> None:
    """D-026: blank lines are not particles and do not break parsing."""
    path = tmp_path / "c.txt"
    path.write_text("2\n1 0 0 0\n0 1 0 0\n0.1 0.2 0 0\n0.3 0.4 0 0\n\n\n")
    pattern = io.read_cco(path)
    assert pattern.n_points == 2
    # edge: blank line between lattice and points, whitespace-only line
    path.write_text("2\n1 0 0 0\n0 1 0 0\n\n0.1 0.2 0 0\n   \n0.3 0.4 0 0\n")
    assert io.read_cco(path).n_points == 2


# ============================================================
# D-027: CCO round trip must be lossless
# ============================================================


def test_regression_D027_cco_full_precision(tmp_path: Path) -> None:
    """D-027: coordinates survive a write/read round trip bit for bit."""
    pattern = PointPattern([[0.1234567891, 0.5]], Box.cubic(1.0, 2))
    back = io.read_cco(io.write_cco(pattern, tmp_path / "c.txt"))
    assert back.positions[0, 0] == 0.1234567891
    # edge: values needing all 17 significant digits
    tricky = PointPattern([[1 / 3, np.nextafter(0.5, 1.0)]], Box.cubic(np.pi, 2))
    back = io.read_cco(io.write_cco(tricky, tmp_path / "d.txt"))
    np.testing.assert_array_equal(back.positions, tricky.positions)
    np.testing.assert_array_equal(back.box.lattice, tricky.box.lattice)


@given(patterns())
@example(PointPattern([[0.1234567891, 0.5]], Box.cubic(1.0, 2)))
def test_regression_D027_round_trip_property(pattern: PointPattern) -> None:
    """D-027 property: write_cco -> read_cco is the identity for any pattern."""
    with tempfile.TemporaryDirectory() as tmp:
        back = io.read_cco(io.write_cco(pattern, Path(tmp) / "c.txt"))
    np.testing.assert_array_equal(back.positions, pattern.positions)
    np.testing.assert_array_equal(back.box.lattice, pattern.box.lattice)


# ============================================================
# D-028: morsefire reader in 3D and with missing files
# ============================================================


def _write_morse(root: Path, lengths: list[float], n: int) -> None:
    root.mkdir()
    d = len(lengths)
    (root / "boxSize.dat").write_text(" ".join(map(str, lengths)) + "\n")
    np.savetxt(root / "positions.dat", np.random.default_rng(1).random((n, d)))
    (root / "scalars.dat").write_text(f"nDim\t{d}\nnumParticles\t{n}\n")


def test_regression_D028_morsefire_3d(tmp_path: Path) -> None:
    """D-028: a 3D box gets all three side lengths (non-singular)."""
    _write_morse(tmp_path / "run", [1.0, 2.0, 3.0], 4)
    pattern = io.read_morsefire(tmp_path / "run")
    np.testing.assert_allclose(pattern.box.lattice, np.diag([1.0, 2.0, 3.0]))
    assert pattern.box.volume == pytest.approx(6.0)


def test_regression_D028_morsefire_missing_file(tmp_path: Path) -> None:
    """D-028: a missing input file raises FileNotFoundError naming the file."""
    _write_morse(tmp_path / "run", [1.0, 2.0], 4)
    (tmp_path / "run" / "positions.dat").unlink()
    with pytest.raises(FileNotFoundError, match=r"positions\.dat"):
        io.read_morsefire(tmp_path / "run")
    # edge: a decoy file whose name contains the real name is not picked up
    (tmp_path / "run" / "old_positions.dat").write_text("0 0\n")
    with pytest.raises(FileNotFoundError, match=r"positions\.dat"):
        io.read_morsefire(tmp_path / "run")
