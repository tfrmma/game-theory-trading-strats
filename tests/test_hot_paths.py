"""
Tests for hot_paths.py (the Cython/pure-Python auto-selecting wrapper).
compute_ofi is the function queue_warfare.py and info_asymmetry.py were
reimplementing by hand before this session's fix these tests pin its
behavior so a future change can't silently break either caller.
"""
from __future__ import annotations

import numpy as np
import pytest

from hot_paths import compute_ofi, backend


def test_backend_returns_a_known_string():
    assert backend() in ("cython", "python")


def test_compute_ofi_matches_manual_sum():
    signed_sizes = np.array([1.0, -0.5, 2.0, -1.5, 0.3], dtype=np.float64)
    assert compute_ofi(signed_sizes) == pytest.approx(sum(signed_sizes))


def test_compute_ofi_empty_array_is_zero():
    assert compute_ofi(np.array([], dtype=np.float64)) == 0.0


def test_compute_ofi_all_buys_is_positive():
    signed_sizes = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    assert compute_ofi(signed_sizes) > 0


def test_compute_ofi_all_sells_is_negative():
    signed_sizes = np.array([-1.0, -2.0, -0.5], dtype=np.float64)
    assert compute_ofi(signed_sizes) < 0


def test_compute_ofi_balanced_flow_is_near_zero():
    signed_sizes = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float64)
    assert compute_ofi(signed_sizes) == pytest.approx(0.0)


def test_compute_ofi_returns_python_float_not_numpy_scalar():
    # queue_warfare.py/info_asymmetry.py store this straight into a deque and
    # sum it later a numpy scalar would work too, but a plain float is the
    # documented contract and avoids surprises mixing with regular floats.
    result = compute_ofi(np.array([1.0, 2.0], dtype=np.float64))
    assert isinstance(result, float)
