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
from pathlib import Path

# ----- third party -----
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from npst_strategies import boxes

# ----- local modules -----
import npointstats
from npointstats.cli import main

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


# ============================================================
# D-047 / D-048: results must not depend on the choice of basis
# ============================================================

SHEARED = np.array([[30.0, 0.0], [27.0, 3.0]])
SHEARED_REDUCED = np.array([[-3.0, 3.0], [30.0, 0.0]])


def _pattern(lattice: np.ndarray, n: int, seed: int) -> "npointstats.PointPattern":
    box = npointstats.Box(lattice)
    s = np.random.default_rng(seed).random((n, box.dimension))
    return npointstats.PointPattern(box.to_cartesian(s), box)


def test_regression_D047_default_dk_keeps_smallest_k() -> None:
    """D-047: the default shell width is the shortest |k|; nothing falls below bin 0."""
    sk = npointstats.structure_factor(_pattern(SHEARED, 500, 0))
    res = sk.radial_average()
    assert res.metadata["dk"] == pytest.approx(sk.k.min())
    assert res.counts.sum() == np.sum(sk.k < res.edges[-1])
    # edge: an ensemble mixing the sheared cell with a square cell drops nothing
    ens = npointstats.Ensemble(
        [_pattern(SHEARED, 500, 1), _pattern(30 * np.eye(2), 500, 2)]
    )
    binned = ens.structure_factor()
    # the finest member spacing wins: 2 pi / 30 for the square cell
    assert binned.metadata["dk"] == pytest.approx(min(sk.k.min(), 2 * np.pi / 30))


def test_regression_D048_limits_independent_of_basis() -> None:
    """D-048: equivalent bases give the same radius limit and mode grid."""
    a, b = npointstats.Box(SHEARED), npointstats.Box(SHEARED_REDUCED)
    assert a.inscribed_radius == pytest.approx(b.inscribed_radius)
    assert a.inscribed_radius > 2.0  # was 1.5 for the sheared basis
    # both reduced bases span the same lattice (integer unimodular transform)
    t = a.reduced_lattice @ np.linalg.inv(b.reduced_lattice)
    np.testing.assert_allclose(t, np.rint(t), atol=1e-9)
    assert abs(abs(np.linalg.det(t)) - 1.0) < 1e-9


@st.composite
def unimodular(draw: st.DrawFn, d: int) -> np.ndarray:
    """Product of a few integer shears and swaps (determinant +-1)."""
    u = np.eye(d, dtype=np.int64)
    for _ in range(draw(st.integers(0, 4))):
        i, j = draw(st.integers(0, d - 1)), draw(st.integers(0, d - 1))
        e = np.eye(d, dtype=np.int64)
        if i == j:
            e[i, i] = -1
        else:
            e[i, j] = draw(st.integers(-3, 3))
        u = e @ u
    return u


@given(
    st.integers(1, 3).flatmap(lambda d: st.tuples(boxes(dimension=d), unimodular(d))),
    st.integers(0, 2**16),
)
@example((npointstats.Box(SHEARED_REDUCED), np.array([[1, 0], [-1, 1]])), 0)
@settings(max_examples=60)
def test_regression_D048_change_of_basis_invariance(case: tuple, seed: int) -> None:
    """D-048 property: a unimodular change of basis changes no statistic."""
    box, u = case
    other = npointstats.Box(u.astype(float) @ box.lattice)
    s = np.random.default_rng(seed).random((12, box.dimension))
    p1 = npointstats.PointPattern(box.to_cartesian(s), box)
    p2 = npointstats.PointPattern(p1.positions, other)
    assert other.inscribed_radius == pytest.approx(box.inscribed_radius, rel=1e-9)
    k_max = 3.0 * float(np.max(np.linalg.norm(box.reduced_reciprocal, axis=1)))
    a = npointstats.structure_factor(p1, k_max=k_max)
    b = npointstats.structure_factor(p2, k_max=k_max)
    assert a.values.size == b.values.size
    np.testing.assert_allclose(
        np.sort(a.values), np.sort(b.values), rtol=1e-8, atol=1e-8
    )
    edges = np.linspace(0.0, box.inscribed_radius, 5)
    np.testing.assert_array_equal(
        npointstats.pairs.pair_counts(p1, edges),
        npointstats.pairs.pair_counts(p2, edges),
    )


# ============================================================
# D-049 .. D-056: I/O, CLI, and API consistency
# ============================================================


def _write_members(root: Path, names: list[str], n: int = 100) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for name in names:
        pattern = npointstats.PointPattern(rng.random((n, 2)), np.eye(2))
        npointstats.write_cco(pattern, root / name)
    return root


def test_regression_D049_raw_sk_names_kept(tmp_path: Path) -> None:
    """D-049: dotted member names give distinct raw S(k) files."""
    root = _write_members(tmp_path / "p", ["cco__phi0.45", "cco__phi0.50"])
    out = tmp_path / "o"
    assert main(["-d", str(root), "-s", "sf", "--raw-sk", "-o", str(out), "-q"]) == 0
    names = sorted(p.name for p in (out / "sk_raw").iterdir())
    assert names == ["cco__phi0.45.npz", "cco__phi0.50.npz"]
    # edge (D-056): raw files use the same k_max as sf.csv
    _, _, meta = npointstats.load_table(out / "sf.csv")
    raw = np.load(out / "sk_raw" / names[0])
    assert float(raw["k_max"]) == pytest.approx(meta["k_max"])


def test_regression_D050_single_member_semantics(tmp_path: Path) -> None:
    """D-050: one-member ensembles report member counts and keep g2 = 0 bins."""
    grid = np.stack(np.meshgrid(np.arange(10), np.arange(10)), -1).reshape(-1, 2)
    root = tmp_path / "one"
    root.mkdir()
    lattice = npointstats.PointPattern(
        grid.astype(float), npointstats.Box.cubic(10.0, 2)
    )
    npointstats.write_cco(lattice, root / "lattice.txt")
    ens = npointstats.Ensemble.from_files(root)
    # dr = 0.3 keeps lattice shells (1, sqrt 2, 2, ...) off the bin edges
    g2 = ens.pair_correlation(dr=0.3)
    assert g2.metadata["n_members"] == 1
    np.testing.assert_array_equal(g2.counts, np.ones(g2.mean.size, dtype=int))
    out = tmp_path / "o"
    assert main(["-d", str(root), "-s", "g2", "--dr", "0.3", "-o", str(out), "-q"]) == 0
    data, _, _ = npointstats.load_table(out / "g2.csv")
    assert data.shape[0] == g2.mean.size == 16
    # the empty region r < 0.9 is written as zeros, not dropped
    np.testing.assert_array_equal(data[:3, 4], 0.0)
    assert data[3, 4] > 0  # nearest-neighbour shell r = 1 in [0.9, 1.2)


def test_regression_D051_csv_header_row(tmp_path: Path) -> None:
    """D-051: the first non-comment line is the column header (pandas-compatible)."""
    import csv

    pattern = npointstats.PointPattern(
        np.random.default_rng(0).random((50, 2)), np.eye(2)
    )
    path = npointstats.save_result(
        npointstats.pair_correlation(pattern, dr=0.05), tmp_path / "g2.csv"
    )
    rows = [
        r
        for r in csv.reader(path.read_text().splitlines())
        if r and not r[0].startswith("#")
    ]
    assert rows[0] == ["r_lo", "r_hi", "r", "r_mean", "g2", "sem", "count"]
    assert len(rows) - 1 == 10
    data, columns, _ = npointstats.load_table(path)
    assert data.shape == (10, 7)
    assert columns == rows[0]


def test_regression_D053_width_larger_than_range() -> None:
    """D-053: a bin wider than the requested range is an error, not a wider range."""
    pattern = npointstats.PointPattern(
        np.random.default_rng(0).random((50, 2)), np.eye(2)
    )
    with pytest.raises(ValueError, match="bin width"):
        npointstats.pair_correlation(pattern, dr=0.4, r_max=0.3)
    with pytest.raises(ValueError, match="bin width"):
        npointstats.binning.uniform_edges(0.4, 0.3)
    # edge: width equal to the range gives exactly one bin
    np.testing.assert_allclose(npointstats.binning.uniform_edges(0.3, 0.3), [0.0, 0.3])


def test_regression_D054_fractional_only_for_dnv_lstj(tmp_path: Path) -> None:
    """D-054: --fractional with a format that ignores it is rejected."""
    root = _write_members(tmp_path / "p", ["a.txt"])
    assert main(["-d", str(root), "-p", "cco", "--fractional", "-s", "g2", "-q"]) == 1


def test_regression_D055_morsefire_missing_keys(tmp_path: Path) -> None:
    """D-055: an incomplete scalars.dat raises ValueError naming the key."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "boxSize.dat").write_text("1.0\n1.0\n")
    np.savetxt(run / "positions.dat", np.random.default_rng(0).random((3, 2)))
    (run / "scalars.dat").write_text("numParticles\t3\n")
    with pytest.raises(ValueError, match="nDim"):
        npointstats.io.read_morsefire(run)


def test_regression_D056_simple_reader_column_selection(tmp_path: Path) -> None:
    """D-056: read_simple(columns=...) selects coordinate columns by index."""
    path = tmp_path / "pts.txt"
    path.write_text("7 0.1 0.2\n8 0.3 0.4\n")
    pattern = npointstats.io.read_simple(path, np.eye(2), columns=(1, 2))
    np.testing.assert_allclose(pattern.positions, [[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="columns"):
        npointstats.io.read_simple(path, np.eye(2), columns=(1,))
    # edge: edges that differ only at tiny scale are not averaged together
    p = npointstats.PointPattern(np.random.default_rng(1).random((30, 2)), np.eye(2))
    a = npointstats.pair_correlation(p, dr=1e-9 * 1e7, r_max=0.3)
    b = npointstats.BinnedResult(
        a.edges * (1 + 1e-10), a.x_mean, a.mean, a.sem, a.counts
    )
    with pytest.raises(ValueError, match="same bin edges"):
        npointstats.ensemble.average_binned([a, b])
