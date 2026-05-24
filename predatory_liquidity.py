from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Deque

import numpy as np

from init import OrderBook, Trade, Signal, Side, SignalStrength, simulate_order_book, simulate_trade_tape

logger = logging.getLogger(__name__)


@dataclass
class LiquidityCluster:
    price: float
    cluster_type: str          # "equal_highs" | "equal_lows"
    touch_count: int
    estimated_resting_size: float
    first_identified_ts: float
    last_touch_ts: float
    active: bool = True

    @property
    def age_s(self) -> float:
        return time.time() - self.first_identified_ts

    def __str__(self) -> str:
        return f"LiqCluster[{self.cluster_type}] price={self.price:.1f} touches={self.touch_count}"


@dataclass
class SweepEvent:
    target_cluster: LiquidityCluster
    direction: Side
    entry_price: float
    sweep_volume: float
    cvd_at_start: float
    cvd_at_peak: float
    oi_change_pct: float
    timestamp: float
    is_exhausted: bool = False


class EqualLevelDetector:
    """
    Detects equal highs/lows — primary resting stop-loss zones.
    Cluster strength grows with touch count; more touches = more stop fuel.
    """

    def __init__(
        self,
        tolerance_pct: float = 0.0005,
        min_touches: int = 2,
        lookback_ticks: int = 200,
        local_extreme_window: int = 5,
    ) -> None:
        self.tolerance_pct        = tolerance_pct
        self.min_touches          = min_touches
        self.lookback_ticks       = lookback_ticks
        self.local_extreme_window = local_extreme_window

        self._mid_history: Deque[float] = deque(maxlen=lookback_ticks)
        self._ts_history:  Deque[float] = deque(maxlen=lookback_ticks)
        self._active_clusters: List[LiquidityCluster] = []

    def update(self, book: OrderBook) -> List[LiquidityCluster]:
        self._mid_history.append(book.mid)
        self._ts_history.append(book.timestamp)

        if len(self._mid_history) < self.local_extreme_window * 2 + 1:
            return []

        self._detect_equal_levels()
        return [c for c in self._active_clusters if c.active]

    def _detect_equal_levels(self) -> None:
        mids = np.array(self._mid_history)
        n, w = len(mids), self.local_extreme_window

        local_highs: List[Tuple[int, float]] = []
        local_lows:  List[Tuple[int, float]] = []

        for i in range(w, n - w):
            window = mids[i - w: i + w + 1]
            if mids[i] == window.max():
                local_highs.append((i, mids[i]))
            if mids[i] == window.min():
                local_lows.append((i, mids[i]))

        self._identify_clusters(local_highs, "equal_highs")
        self._identify_clusters(local_lows,  "equal_lows")

    def _identify_clusters(self, extremes: List[Tuple[int, float]], cluster_type: str) -> None:
        if len(extremes) < self.min_touches:
            return

        used = set()
        for i, (_, price_i) in enumerate(extremes):
            if i in used:
                continue
            group = [i]
            for j, (_, price_j) in enumerate(extremes):
                if j in used or j == i:
                    continue
                if abs(price_j - price_i) / price_i < self.tolerance_pct:
                    group.append(j)
                    used.add(j)
            used.add(i)

            if len(group) < self.min_touches:
                continue

            cluster_price = float(np.mean([extremes[k][1] for k in group]))
            existing = self._find_existing_cluster(cluster_price)
            if existing:
                existing.touch_count = max(existing.touch_count, len(group))
                existing.last_touch_ts = time.time()
                existing.estimated_resting_size = existing.touch_count * 5.0
            else:
                self._active_clusters.append(LiquidityCluster(
                    price=cluster_price,
                    cluster_type=cluster_type,
                    touch_count=len(group),
                    estimated_resting_size=len(group) * 5.0,
                    first_identified_ts=time.time(),
                    last_touch_ts=time.time(),
                ))

    def _find_existing_cluster(self, price: float) -> Optional[LiquidityCluster]:
        for c in self._active_clusters:
            if abs(c.price - price) / price < self.tolerance_pct and c.active:
                return c
        return None

    def invalidate_cluster(self, price: float) -> None:
        for c in self._active_clusters:
            if abs(c.price - price) / price < self.tolerance_pct:
                c.active = False
                logger.info("Cluster swept: %s", c)


class CVDTracker:
    """Cumulative Volume Delta. Divergence from price = hidden accumulation/distribution."""

    def __init__(self, window: int = 200) -> None:
        self._deltas: Deque[float] = deque(maxlen=window)
        self._cvd_snapshots: Deque[float] = deque(maxlen=window)
        self._running_cvd: float = 0.0

    def update_trades(self, trades: List[Trade]) -> float:
        for trade in trades:
            delta = trade.size if trade.side == Side.BUY else -trade.size
            self._deltas.append(delta)
            self._running_cvd += delta
        self._cvd_snapshots.append(self._running_cvd)
        return self._running_cvd

    @property
    def current(self) -> float:
        return self._running_cvd

    @property
    def recent_delta(self) -> float:
        recent = list(self._deltas)[-20:]
        return float(sum(recent)) if recent else 0.0

    def is_diverging_from_price(self, price_up: bool, lookback: int = 50) -> bool:
        snaps = list(self._cvd_snapshots)[-lookback:]
        if len(snaps) < 10:
            return False
        return price_up != (snaps[-1] > snaps[0])


class PredatoryLiquidityStrategy:
    """
    Two-mode stop-hunt strategy:
      JOIN  — enter in sweep direction before the cluster is hit, ride the stop cascade.
      FADE  — enter against sweep direction once volume spike + CVD reversal confirm exhaustion.
    """

    def __init__(
        self,
        join_distance_pct: float = 0.003,
        fade_wick_threshold: float = 0.0015,
        min_cluster_touches: int = 3,
        min_sweep_volume_multiple: float = 3.0,
        target_reversion_pct: float = 0.004,
        stop_loss_pct: float = 0.002,
    ) -> None:
        self.join_distance_pct        = join_distance_pct
        self.fade_wick_threshold      = fade_wick_threshold
        self.min_cluster_touches      = min_cluster_touches
        self.min_sweep_volume_multiple = min_sweep_volume_multiple
        self.target_reversion_pct     = target_reversion_pct
        self.stop_loss_pct            = stop_loss_pct

        self.cluster_detector = EqualLevelDetector()
        self.cvd_tracker      = CVDTracker()
        self._baseline_volume: Deque[float] = deque(maxlen=50)
        self._mid_history: Deque[float] = deque(maxlen=100)

    def update(self, book: OrderBook, recent_trades: List[Trade]) -> Optional[Signal]:
        self.cluster_detector.update(book)
        self.cvd_tracker.update_trades(recent_trades)

        tick_volume = sum(t.size for t in recent_trades)
        self._baseline_volume.append(tick_volume)
        self._mid_history.append(book.mid)

        strong_clusters = [
            c for c in self.cluster_detector._active_clusters
            if c.touch_count >= self.min_cluster_touches and c.active
        ]
        if not strong_clusters:
            return None

        baseline_vol = float(np.mean(self._baseline_volume)) if self._baseline_volume else 1.0

        for cluster in strong_clusters:
            signal = self._evaluate_cluster(book, cluster, tick_volume, baseline_vol)
            if signal:
                return signal

        return None

    def _evaluate_cluster(
        self,
        book: OrderBook,
        cluster: LiquidityCluster,
        tick_volume: float,
        baseline_vol: float,
    ) -> Optional[Signal]:
        mid      = book.mid
        dist_pct = abs(mid - cluster.price) / mid
        price_up = mid > (list(self._mid_history)[-5] if len(self._mid_history) >= 5 else mid)

        approaching_high = (
            cluster.cluster_type == "equal_highs"
            and mid < cluster.price
            and dist_pct < self.join_distance_pct
            and self.cvd_tracker.recent_delta > 0
        )
        approaching_low = (
            cluster.cluster_type == "equal_lows"
            and mid > cluster.price
            and dist_pct < self.join_distance_pct
            and self.cvd_tracker.recent_delta < 0
        )

        if approaching_high:
            return self._make_signal(
                Side.BUY, mid * 1.00005,
                cluster.price * (1 + self.target_reversion_pct),
                mid * (1 - self.stop_loss_pct),
                cluster, SignalStrength.STRONG,
                f"JOIN HUNT: approaching equal_highs @ {cluster.price:.1f} ({cluster.touch_count} touches) "
                f"CVD={self.cvd_tracker.recent_delta:.2f}",
            )

        if approaching_low:
            return self._make_signal(
                Side.SELL, mid * 0.99995,
                cluster.price * (1 - self.target_reversion_pct),
                mid * (1 + self.stop_loss_pct),
                cluster, SignalStrength.STRONG,
                f"JOIN HUNT: approaching equal_lows @ {cluster.price:.1f} ({cluster.touch_count} touches) "
                f"CVD={self.cvd_tracker.recent_delta:.2f}",
            )

        volume_spike = tick_volume > baseline_vol * self.min_sweep_volume_multiple
        wick_above   = (
            cluster.cluster_type == "equal_highs"
            and mid > cluster.price * (1 + self.fade_wick_threshold)
            and volume_spike
            and self.cvd_tracker.is_diverging_from_price(price_up=True)
        )
        wick_below   = (
            cluster.cluster_type == "equal_lows"
            and mid < cluster.price * (1 - self.fade_wick_threshold)
            and volume_spike
            and self.cvd_tracker.is_diverging_from_price(price_up=False)
        )

        if wick_above:
            self.cluster_detector.invalidate_cluster(cluster.price)
            return self._make_signal(
                Side.SELL, mid * 0.99995,
                cluster.price * (1 - self.target_reversion_pct),
                mid * (1 + self.stop_loss_pct * 0.75),
                cluster, SignalStrength.EXTREME,
                f"FADE POST-SWEEP: wicked {dist_pct*100:.2f}% above equal_highs @ {cluster.price:.1f} "
                f"vol={tick_volume/baseline_vol:.1f}x CVD diverging",
            )

        if wick_below:
            self.cluster_detector.invalidate_cluster(cluster.price)
            return self._make_signal(
                Side.BUY, mid * 1.00005,
                cluster.price * (1 + self.target_reversion_pct),
                mid * (1 - self.stop_loss_pct * 0.75),
                cluster, SignalStrength.EXTREME,
                f"FADE POST-SWEEP: wicked below equal_lows @ {cluster.price:.1f} "
                f"vol={tick_volume/baseline_vol:.1f}x CVD reversing",
            )

        return None

    def _make_signal(
        self,
        direction: Side,
        entry: float,
        target: float,
        stop: float,
        cluster: LiquidityCluster,
        strength: SignalStrength,
        rationale: str,
    ) -> Signal:
        return Signal(
            direction=direction,
            strength=strength,
            confidence=min(1.0, cluster.touch_count / 5.0),
            entry_price=entry,
            target_price=target,
            stop_price=stop,
            rationale=rationale,
            invalidation_conditions=[
                "OI expands through cluster (genuine breakout)",
                "CVD stays aligned with sweep 60s after entry",
                "Macro news within 15min of signal",
                "Price fails to retrace 30% of sweep in 5min",
            ],
        )


def simulate_stop_hunt(n_ticks: int = 300, mid: float = 50_000.0) -> None:
    rng           = np.random.default_rng(99)
    strategy      = PredatoryLiquidityStrategy()
    cluster_level = 50_050.0
    current_mid   = 49_900.0

    print("\n" + "="*65)
    print("PREDATORY LIQUIDITY / STOP HUNT SIMULATION")
    print("="*65)

    for i in range(n_ticks):
        if i < 150:
            current_mid += (cluster_level - current_mid) * 0.02 + rng.normal(0, 15)
            current_mid  = max(49_700.0, min(current_mid, cluster_level - 10))
        elif i < 180:
            current_mid += rng.normal(20, 5)
        else:
            current_mid += rng.normal(-15, 8)

        book   = simulate_order_book(mid=current_mid, rng=rng)
        trades = simulate_trade_tape(
            n_trades=rng.integers(3, 15), mid=current_mid, sigma_per_trade=5.0,
            informed_fraction=0.3 if 150 <= i < 180 else 0.1, rng=rng,
        )
        signal = strategy.update(book, trades)

        if signal:
            print(f"\n[Tick {i}] MID={current_mid:.1f}")
            print(f"  {signal.direction.value.upper()} @ {signal.entry_price:.2f} | "
                  f"target={signal.target_price:.2f} | stop={signal.stop_price:.2f} | "
                  f"R/R={signal.risk_reward:.2f}x")
            print(f"  {signal.rationale}")

        if i % 50 == 0:
            clusters = [c for c in strategy.cluster_detector._active_clusters if c.active]
            print(f"[Tick {i:3d}] mid={current_mid:.1f} clusters={len(clusters)} cvd={strategy.cvd_tracker.current:.2f}")

    print("\n" + "="*65 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    simulate_stop_hunt()