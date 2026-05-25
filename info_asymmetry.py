from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from init import Trade, OrderBook, Side, Signal, SignalStrength, simulate_order_book, simulate_trade_tape

logger = logging.getLogger(__name__)


@dataclass
class VPINBucket:
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0

    @property
    def imbalance(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return abs(self.buy_volume - self.sell_volume) / self.total_volume


class VPINCalculator:
    """
    Volume-synchronized PIN (Easley, Lopez de Prado & O'Hara 2012).
    VPIN ∈ [0,1]: average order imbalance across equal-volume buckets.
    Leads adverse price moves — spikes before the damage lands.
    """

    REGIME_THRESHOLDS = [(0.20, "SAFE"), (0.40, "CAUTION"), (0.60, "TOXIC")]

    def __init__(
        self,
        bucket_size: float = 100.0,
        n_buckets: int = 50,
        min_buckets_for_signal: int = 10,
    ) -> None:
        self.bucket_size = bucket_size
        self.n_buckets   = n_buckets
        self.min_buckets  = min_buckets_for_signal

        self._current_bucket = VPINBucket()
        self._completed_buckets: Deque[VPINBucket] = deque(maxlen=n_buckets)
        self._remaining: float = bucket_size

    def resize_bucket(self, new_size: float, resize_threshold: float = 0.20) -> bool:
        """
        Resize bucket_size when ADV-derived target deviates by more than threshold.
        Resets the current partial bucket to avoid a corrupted imbalance reading.
        Returns True if a resize actually happened.
        """
        if abs(new_size - self.bucket_size) / max(self.bucket_size, 1e-9) > resize_threshold:
            self.bucket_size      = new_size
            self._current_bucket  = VPINBucket()
            self._remaining       = new_size
            return True
        return False

    def update(self, trades: List[Trade]) -> Optional[float]:
        for trade in trades:
            self._fill_bucket(trade)
        return self.current_vpin

    def _fill_bucket(self, trade: Trade) -> None:
        remaining = trade.size
        while remaining > 0:
            fill = min(remaining, self._remaining)
            if trade.side == Side.BUY:
                self._current_bucket.buy_volume += fill
            else:
                self._current_bucket.sell_volume += fill
            self._current_bucket.total_volume += fill
            self._remaining -= fill
            remaining -= fill

            if self._remaining <= 0:
                self._completed_buckets.append(self._current_bucket)
                self._current_bucket = VPINBucket()
                self._remaining = self.bucket_size

    @property
    def current_vpin(self) -> Optional[float]:
        buckets = list(self._completed_buckets)
        if len(buckets) < self.min_buckets:
            return None
        return float(np.mean([b.imbalance for b in buckets]))

    def regime(self, vpin: float) -> str:
        for threshold, label in self.REGIME_THRESHOLDS:
            if vpin < threshold:
                return label
        return "EXTREME_TOXICITY"

    @property
    def spread_multiplier(self) -> float:
        v = self.current_vpin
        if v is None or v < 0.20: return 1.0
        if v < 0.40: return 1.3
        if v < 0.60: return 2.0
        return 5.0


class KyleLambdaEstimator:
    """
    OLS estimate of price impact: ΔP = α + λ·OFI + ε
    High λ → thin book or persistent informed flow. Scale spreads accordingly.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._ofi_history: Deque[float] = deque(maxlen=window)
        self._dp_history:  Deque[float] = deque(maxlen=window)
        self._lambda: float = 0.0
        self._alpha: float  = 0.0
        self._r_squared: float = 0.0

    def update(self, ofi: float, dp: float) -> Optional[float]:
        self._ofi_history.append(ofi)
        self._dp_history.append(dp)
        if len(self._ofi_history) < 20:
            return None
        return self._fit()

    def _fit(self) -> float:
        X = np.column_stack([np.ones(len(self._ofi_history)), np.array(self._ofi_history)])
        y = np.array(self._dp_history)
        try:
            coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self._alpha, self._lambda = coefs[0], coefs[1]
            y_hat = X @ coefs
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            self._r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except np.linalg.LinAlgError:
            pass
        return self._lambda

    @property
    def lambda_estimate(self) -> float:
        return max(0.0, self._lambda)

    @property
    def r_squared(self) -> float:
        return self._r_squared

    def adverse_selection_cost(self, expected_ofi: float) -> float:
        return self.lambda_estimate * abs(expected_ofi)



class RollingADV:
    """
    Rolling Average Daily Volume over a configurable window.

    Maintains a timestamped deque of per-tick volumes and exposes the total
    volume observed in the last `window_h` hours. VPINCalculator uses this
    to resize buckets so that a fixed number of buckets spans the observed
    flow regardless of how quiet or hectic the market is.

    Bucket sizing:
        bucket_size = adv * adv_fraction
    where adv_fraction = 1 / n_buckets_per_adv_window, e.g. 1/50 means
    each bucket represents 2% of the rolling daily volume.

    If less than `min_obs` ticks have been seen, returns the provided
    fallback so VPIN can still run during the warm-up period.
    """

    def __init__(
        self,
        window_h: float = 24.0,
        n_buckets_per_window: int = 50,
        min_obs: int = 20,
        fallback_bucket_size: float = 100.0,
    ) -> None:
        self.window_s             = window_h * 3600.0
        self.n_buckets_per_window = n_buckets_per_window
        self.min_obs              = min_obs
        self.fallback_bucket_size = fallback_bucket_size

        self._vol_history: Deque[tuple] = deque()   # (timestamp, volume)

    def update(self, trades: List[Trade]) -> float:
        """Ingest new trades and return the current recommended bucket_size."""
        tick_vol = sum(t.size for t in trades)
        now      = time.time()
        self._vol_history.append((now, tick_vol))
        self._expire_old(now)
        return self.bucket_size

    def _expire_old(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._vol_history and self._vol_history[0][0] < cutoff:
            self._vol_history.popleft()

    @property
    def rolling_volume(self) -> float:
        return sum(v for _, v in self._vol_history)

    @property
    def bucket_size(self) -> float:
        if len(self._vol_history) < self.min_obs:
            return self.fallback_bucket_size
        return max(self.fallback_bucket_size * 0.1, self.rolling_volume / self.n_buckets_per_window)


class FlowToxicityClassifier:
    """
    Composite real-time toxicity: VPIN (60%) + Kyle's λ (40%).
    Outputs spread multiplier and directional signal when informed flow is persistent.
    """

    def __init__(
        self,
        vpin_toxic_threshold: float = 0.45,
        lambda_toxic_threshold: float = 0.05,
        cvd_momentum_threshold: float = 500.0,
        adv_window_h: float = 24.0,
        adv_buckets_per_window: int = 50,
    ) -> None:
        self.adv_tracker = RollingADV(
            window_h=adv_window_h,
            n_buckets_per_window=adv_buckets_per_window,
        )
        self.vpin_calc   = VPINCalculator()
        self.lambda_est  = KyleLambdaEstimator()
        self.vpin_toxic_threshold   = vpin_toxic_threshold
        self.lambda_toxic_threshold = lambda_toxic_threshold
        self.cvd_momentum_threshold = cvd_momentum_threshold

        self._prev_mid: Optional[float] = None
        self._cumulative_ofi: float = 0.0
        self._ofi_window: Deque[float] = deque(maxlen=20)

    def update(self, book: OrderBook, trades: List[Trade]) -> dict:
        # Recompute bucket_size from rolling ADV before filling buckets
        target_bucket_size = self.adv_tracker.update(trades)
        resized = self.vpin_calc.resize_bucket(target_bucket_size)
        if resized:
            logger.debug("VPIN bucket resized → %.2f (ADV-derived)", target_bucket_size)

        vpin = self.vpin_calc.update(trades)

        ofi = sum(t.signed_size for t in trades)
        self._ofi_window.append(ofi)
        self._cumulative_ofi += ofi

        if self._prev_mid is not None:
            self.lambda_est.update(ofi, book.mid - self._prev_mid)
        self._prev_mid = book.mid

        vpin_score   = min(1.0, (vpin or 0.0) / 0.8)
        lambda_score = min(1.0, self.lambda_est.lambda_estimate / (self.lambda_toxic_threshold * 2))
        toxicity     = 0.6 * vpin_score + 0.4 * lambda_score

        directional: Optional[Side] = None
        if (vpin or 0) > self.vpin_toxic_threshold:
            recent_ofi = float(np.sum(list(self._ofi_window)))
            if recent_ofi > self.cvd_momentum_threshold:
                directional = Side.BUY
            elif recent_ofi < -self.cvd_momentum_threshold:
                directional = Side.SELL

        regime = self.vpin_calc.regime(vpin or 0.0)

        if toxicity > 0.6:
            logger.warning(
                "HIGH TOXICITY: vpin=%.3f λ=%.5f regime=%s spread_mult=%.1fx",
                vpin or 0, self.lambda_est.lambda_estimate, regime,
                self.vpin_calc.spread_multiplier,
            )

        return {
            "vpin":               vpin,
            "lambda":             self.lambda_est.lambda_estimate,
            "r_squared":          self.lambda_est.r_squared,
            "toxicity_score":     toxicity,
            "spread_multiplier":  self.vpin_calc.spread_multiplier,
            "regime":             regime,
            "directional_signal": directional,
            "cumulative_ofi":     self._cumulative_ofi,
            "bucket_size":        self.vpin_calc.bucket_size,
            "adv":                self.adv_tracker.rolling_volume,
        }


def simulate_informed_flow(
    n_episodes: int = 3,
    n_trades_per_episode: int = 150,
    mid: float = 50_000.0,
) -> None:
    rng = np.random.default_rng(7)
    classifier = FlowToxicityClassifier()

    print("\n" + "="*65)
    print("INFO ASYMMETRY / VPIN / KYLE'S LAMBDA")
    print("="*65)

    for episode in range(n_episodes * 2):
        is_informed  = episode % 2 == 1
        informed_pct = 0.40 if is_informed else 0.05
        label        = "INFORMED" if is_informed else "NOISE"

        print(f"\n{'─'*50}")
        print(f"Episode {episode+1}: {label}")
        print(f"{'─'*50}")

        for tick in range(n_trades_per_episode // 10):
            book   = simulate_order_book(mid=mid, rng=rng)
            trades = simulate_trade_tape(10, mid, 5.0, informed_pct, rng)
            state  = classifier.update(book, trades)

            if tick % 5 == 0 or state["toxicity_score"] > 0.5:
                signal_str = f" | SIGNAL: {state['directional_signal'].value}" if state["directional_signal"] else ""
                print(
                    f"  tick={tick:3d} | vpin={state['vpin'] or 0:.3f} | "
                    f"λ={state['lambda']:.5f} | toxicity={state['toxicity_score']:.2f} | "
                    f"regime={state['regime']:<18s} | spread_mult={state['spread_multiplier']:.1f}x | "
                    f"bucket={state['bucket_size']:.2f} adv={state['adv']:.1f}"
                    + signal_str
                )

            mid += rng.normal(3.0, 1.0) if is_informed else rng.normal(0.0, 5.0)

    print("\n" + "="*65 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")
    simulate_informed_flow()
