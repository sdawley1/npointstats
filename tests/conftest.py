#!/usr/bin/env python3

"""
Shared pytest configuration: Hypothesis profiles and seeded fixtures.

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
from hypothesis import HealthCheck, settings

# ============================================================
# HYPOTHESIS PROFILES
# ============================================================

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci", max_examples=200, deadline=None, derandomize=True, print_blob=True
)
settings.register_profile(
    "thorough",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("dev")

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator, fresh for every test."""
    return np.random.default_rng(20261026)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test inside its own temporary directory (CLI defaults to -o .)."""
    monkeypatch.chdir(tmp_path)
