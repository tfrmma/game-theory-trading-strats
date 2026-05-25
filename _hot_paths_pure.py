# Pure Python fallback. Identical API to _hot_paths.pyx. Use via hot_paths.py.
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def book_imbalance(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    n_levels: int,
) -> float:
    n       = min(n_levels, len(bid_sizes), len(ask_sizes))
    bid_vol = float(bid_sizes[:n].sum())
    ask_vol = float(ask_sizes[:n].sum())
    total   = bid_vol + ask_vol
    return (bid_vol - ask_vol) / total if total > 0 else 0.0


def compute_ofi(signed_sizes: np.ndarray) -> float:
    return float(signed_sizes.sum())


def realized_vol(prices: np.ndarray, window: int) -> float:
    if len(prices) < 2:
        return 0.0
    p = prices[-min(window + 1, len(prices)):]
    lr = np.diff(np.log(np.maximum(p, 1e-12)))
    return float(np.std(lr))


def kyle_lambda_ols(
    ofi_window: np.ndarray,
    dp_window:  np.ndarray,
) -> Tuple[float, float, float]:
    n = len(ofi_window)
    if n < 2:
        return 0.0, 0.0, 0.0
    X = np.column_stack([np.ones(n), ofi_window])
    y = dp_window
    try:
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha, lam = float(coefs[0]), float(coefs[1])
        y_hat  = X @ coefs
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return lam, alpha, r2
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0


def poisson_fill_prob(
    queue_ahead:  float,
    arrival_rate: float,
    horizon_s:    float,
) -> float:
    if arrival_rate <= 0 or horizon_s <= 0:
        return 0.0
    if queue_ahead <= 0:
        return 1.0

    from scipy.stats import poisson as _poisson
    mu = arrival_rate * horizon_s
    k  = queue_ahead

    if k > 50:
        z = (mu - k) / math.sqrt(mu + 1e-9)
        return float(1.0 / (1.0 + math.exp(-1.7 * z)))

    p = float(1.0 - _poisson.cdf(int(math.floor(k)), mu=mu))
    return max(0.0, min(1.0, p))


def fill_vpin_buckets(
    buy_vols:      np.ndarray,
    sell_vols:     np.ndarray,
    bucket_size:   float,
    current_fill:  float,
    current_buy:   float,
    current_sell:  float,
) -> Tuple[List[float], float, float, float]:
    completed: List[float] = []
    fill = current_fill
    buy  = current_buy
    sell = current_sell

    for b_vol, s_vol in zip(buy_vols, sell_vols):
        remaining = b_vol + s_vol
        total_vol = b_vol + s_vol

        while remaining > 1e-12:
            can_fill = bucket_size - fill
            take     = min(remaining, can_fill)

            if total_vol > 0:
                buy  += take * b_vol / total_vol
                sell += take * s_vol / total_vol

            fill      += take
            remaining -= take

            if fill >= bucket_size - 1e-10:
                total = buy + sell
                if total > 0:
                    completed.append(abs(buy - sell) / total)
                fill = buy = sell = 0.0

    return completed, fill, buy, sell


def lot_floor(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step
