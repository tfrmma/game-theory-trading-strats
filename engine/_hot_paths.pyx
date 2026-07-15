# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

cimport cython
from libc.math cimport log, sqrt, exp, fabs, floor
import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


# Order book imbalance

cpdef double book_imbalance(
    double[:] bid_sizes,
    double[:] ask_sizes,
    int n_levels,
) nogil:
    """
    Weighted order flow imbalance ∈ [-1, 1].
    Positive = bid-heavy (buy pressure). Negative = ask-heavy.
    Uses only the first n_levels of each side.
    """
    cdef int i, n
    cdef double bid_vol = 0.0, ask_vol = 0.0, total

    n = min(n_levels, bid_sizes.shape[0], ask_sizes.shape[0])
    for i in range(n):
        bid_vol += bid_sizes[i]
        ask_vol += ask_sizes[i]

    total = bid_vol + ask_vol
    if total == 0.0:
        return 0.0
    return (bid_vol - ask_vol) / total


# Cumulative volume delta (OFI)

cpdef double compute_ofi(double[:] signed_sizes) nogil:
    """
    Sum of signed trade sizes (+ = buy aggression, - = sell aggression).
    Runs in pure C — no Python object creation in the loop.
    """
    cdef int i
    cdef double ofi = 0.0
    for i in range(signed_sizes.shape[0]):
        ofi += signed_sizes[i]
    return ofi


# Realized volatility

cpdef double realized_vol(double[:] prices, int window) nogil:
    """
    Std of log returns over the last `window` prices.
    Returns 0.0 if fewer than 2 prices are available.
    """
    cdef int n, i, actual_window
    cdef double mean = 0.0, var = 0.0, lr
    cdef double[100] log_returns  # stack-allocated, avoids heap for small windows

    n = prices.shape[0]
    if n < 2:
        return 0.0

    actual_window = min(window, n - 1)
    if actual_window > 100:
        actual_window = 100  # guard against stack overflow

    # Compute log returns for the last actual_window steps
    for i in range(actual_window):
        idx = n - actual_window - 1 + i
        if prices[idx] <= 0.0 or prices[idx + 1] <= 0.0:
            log_returns[i] = 0.0
        else:
            log_returns[i] = log(prices[idx + 1] / prices[idx])
        mean += log_returns[i]

    mean /= actual_window
    for i in range(actual_window):
        var += (log_returns[i] - mean) ** 2
    var /= actual_window

    return sqrt(var)


# Kyle's lambda OLS (incremental, O(window) per call)

cpdef tuple kyle_lambda_ols(
    double[:] ofi_window,
    double[:] dp_window,
) nogil:
    """
    OLS: ΔP = α + λ·OFI.
    Returns (lambda, alpha, r_squared).
    Expects equal-length windows of at least 2 observations.
    All arithmetic in C doubles — no Python allocation.
    """
    cdef int n, i
    cdef double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0
    cdef double denom, lam, alpha, y_hat, ss_res = 0.0, ss_tot = 0.0
    cdef double mean_y, r2

    n = ofi_window.shape[0]
    if n < 2:
        return (0.0, 0.0, 0.0)

    for i in range(n):
        sx  += ofi_window[i]
        sy  += dp_window[i]
        sxx += ofi_window[i] * ofi_window[i]
        sxy += ofi_window[i] * dp_window[i]

    denom = n * sxx - sx * sx
    if fabs(denom) < 1e-12:
        return (0.0, 0.0, 0.0)

    lam   = (n * sxy - sx * sy) / denom
    alpha = (sy - lam * sx) / n

    mean_y = sy / n
    for i in range(n):
        y_hat   = alpha + lam * ofi_window[i]
        ss_res += (dp_window[i] - y_hat) ** 2
        ss_tot += (dp_window[i] - mean_y) ** 2

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return (lam, alpha, r2)


# Poisson fill probability

cpdef double poisson_fill_prob(
    double queue_ahead,
    double arrival_rate,
    double horizon_s,
) nogil:
    """
    P(cumulative Poisson arrivals × mean_size ≥ queue_ahead within horizon_s).

    We approximate via: 1 - CDF_Poisson(k-1, mu) where k = queue_ahead / mean_trade_size.
    Here we pass queue_ahead already normalized by mean_trade_size (i.e., k directly).

    Uses the regularized incomplete gamma function approximation:
        P(X ≥ k | mu) ≈ 1 − e^(−mu) × Σ mu^i/i! for i < k

    For k <= 20 we compute the partial sum directly; for k > 20 we use a
    normal approximation (mu large enough for CLT to hold).
    """
    cdef double mu, k, p, term, cum_prob
    cdef int i, k_int

    if arrival_rate <= 0.0 or horizon_s <= 0.0:
        return 0.0

    mu = arrival_rate * horizon_s
    k  = queue_ahead  # already normalized

    if k <= 0.0:
        return 1.0

    if k > 50.0:
        # Normal approximation: Z = (mu - k) / sqrt(mu)
        cdef double z = (mu - k) / sqrt(mu + 1e-9)
        # Φ(z) ≈ 0.5*(1 + erf(z/√2)); use logistic approximation
        return 1.0 / (1.0 + exp(-1.7 * z))

    # Exact Poisson CDF via partial sum
    k_int    = <int>floor(k)
    term     = exp(-mu)
    cum_prob = term
    for i in range(1, k_int + 1):
        term     *= mu / i
        cum_prob += term

    p = 1.0 - cum_prob
    if p < 0.0: p = 0.0
    if p > 1.0: p = 1.0
    return p


# VPIN bucket fill (inner loop only — state managed in Python)

cpdef tuple fill_vpin_buckets(
    double[:] buy_vols,
    double[:] sell_vols,
    double bucket_size,
    double current_fill,
    double current_buy,
    double current_sell,
):
    """
    Fills VPIN buckets from per-trade buy/sell volume arrays.

    Returns:
        (completed_imbalances, final_fill, final_buy, final_sell)
        completed_imbalances: list of |buy - sell| / total for each completed bucket
    """
    cdef int n, i
    cdef double fill, buy, sell, b_vol, s_vol, remaining, total
    cdef list completed = []

    n    = buy_vols.shape[0]
    fill = current_fill
    buy  = current_buy
    sell = current_sell

    for i in range(n):
        b_vol = buy_vols[i]
        s_vol = sell_vols[i]
        remaining = b_vol + s_vol

        while remaining > 0.0:
            can_fill = bucket_size - fill
            take     = remaining if remaining <= can_fill else can_fill

            # Proportionally split into buy/sell
            if b_vol + s_vol > 0.0:
                buy  += take * b_vol / (b_vol + s_vol)
                sell += take * s_vol / (b_vol + s_vol)
            fill      += take
            remaining -= take

            if fill >= bucket_size - 1e-10:
                total = buy + sell
                if total > 0.0:
                    completed.append(fabs(buy - sell) / total)
                fill = 0.0
                buy  = 0.0
                sell = 0.0

    return (completed, fill, buy, sell)


# Lot-size floor (used in CentralRiskManager._shave hot path)

cpdef double lot_floor(double value, double step) nogil:
    """
    Floor `value` to the nearest multiple of `step`.
    Avoids Python math.floor overhead in the shave hot path.
    """
    if step <= 0.0:
        return value
    return floor(value / step) * step
