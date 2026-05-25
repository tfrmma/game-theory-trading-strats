# Auto-selects compiled Cython extension if available, else pure Python fallback.
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from _hot_paths import (
        book_imbalance,
        compute_ofi,
        realized_vol,
        kyle_lambda_ols,
        poisson_fill_prob,
        fill_vpin_buckets,
        lot_floor,
    )
    _BACKEND = "cython"
except ImportError:
    from _hot_paths_pure import (  # type: ignore[no-redef]
        book_imbalance,
        compute_ofi,
        realized_vol,
        kyle_lambda_ols,
        poisson_fill_prob,
        fill_vpin_buckets,
        lot_floor,
    )
    _BACKEND = "python"
    logger.info("hot_paths: Cython extension not found — using pure Python fallback.")

__all__ = [
    "book_imbalance",
    "compute_ofi",
    "realized_vol",
    "kyle_lambda_ols",
    "poisson_fill_prob",
    "fill_vpin_buckets",
    "lot_floor",
    "backend",
]


def backend() -> str:
    return _BACKEND
