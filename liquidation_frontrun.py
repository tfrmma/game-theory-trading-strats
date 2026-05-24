from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from init import OrderBook, Trade, Side, simulate_order_book, simulate_trade_tape

logger = logging.getLogger(__name__)


@dataclass
class LiquidationCluster:
    trigger_price: float
    direction: Side          # SELL = liquidates longs, BUY = liquidates shorts
    estimated_volume: float
    confidence: float
    oi_at_detection: float
    detection_ts: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"LiqCluster[{self.direction.value}] trigger={self.trigger_price:.1f} "
            f"est_vol={self.estimated_volume:.2f} conf={self.confidence:.2f}"
        )


class LiquidationClusterMapper:
    """
    Estimates liquidation clusters from OI + price history.
    When OI spikes during a trend, new leveraged positions are being added.
    Those positions have predictable liquidation prices based on leverage band.
    """

    def __init__(
        self,
        leverage_profiles: List[float] = [5.0, 10.0, 20.0, 50.0],
        window_s: float = 3600.0,
        min_confidence: float = 0.30,
    ) -> None:
        self.leverage_profiles = leverage_profiles
        self.window_s          = window_s
        self.min_confidence    = min_confidence

        self._oi_snapshots: Deque[Tuple[float, float, float]] = deque(maxlen=200)
        self._identified_clusters: List[LiquidationCluster] = []

    def update(self, oi: float, mid: float) -> List[LiquidationCluster]:
        ts = time.time()
        self._oi_snapshots.append((ts, oi, mid))

        if len(self._oi_snapshots) < 10:
            return []

        new_clusters = self._identify_oi_buildup_clusters()
        self._identified_clusters.extend(new_clusters)
        self._identified_clusters = [
            c for c in self._identified_clusters if ts - c.detection_ts < self.window_s
        ]
        return new_clusters

    def _identify_oi_buildup_clusters(self) -> List[LiquidationCluster]:
        snaps = list(self._oi_snapshots)
        if len(snaps) < 20:
            return []

        ois  = np.array([s[1] for s in snaps])
        mids = np.array([s[2] for s in snaps])
        tss  = np.array([s[0] for s in snaps])

        oi_change = np.diff(ois)
        buildup_idx = np.where(oi_change > ois[:-1] * 0.005)[0]

        clusters: List[LiquidationCluster] = []
        for idx in buildup_idx[-10:]:
            entry_price = float(mids[idx])
            oi_added    = float(oi_change[idx])

            for leverage in self.leverage_profiles:
                liq_drop    = 1.0 / leverage
                recency     = (time.time() - float(tss[idx])) / self.window_s
                confidence  = (1 - recency) * min(1.0, oi_added / (ois[idx] + 1e-9) * 20)

                if confidence < self.min_confidence:
                    continue

                clusters.append(LiquidationCluster(
                    trigger_price=entry_price * (1 - liq_drop),
                    direction=Side.SELL,
                    estimated_volume=oi_added * 0.3,
                    confidence=confidence,
                    oi_at_detection=float(ois[idx]),
                ))

        return clusters

    def get_active_clusters(self, mid: float, proximity_pct: float = 0.05) -> List[LiquidationCluster]:
        return [c for c in self._identified_clusters if abs(c.trigger_price - mid) / mid < proximity_pct]


class CascadeDetector:
    """
    Signals cascade start (enter taker) and exhaustion (place maker absorption).
    Start: volume spike + OI drop + liquidation flags.
    Exhaustion: volume decelerating + delta reversal.
    """

    def __init__(
        self,
        volume_spike_multiple: float = 3.0,
        oi_drop_threshold_pct: float = 0.005,
        window_ticks: int = 20,
    ) -> None:
        self.volume_spike_multiple = volume_spike_multiple
        self.oi_drop_threshold_pct = oi_drop_threshold_pct

        self._vol_history:   Deque[float] = deque(maxlen=window_ticks * 5)
        self._oi_history:    Deque[float] = deque(maxlen=window_ticks)
        self._price_history: Deque[float] = deque(maxlen=window_ticks)
        self._delta_history: Deque[float] = deque(maxlen=window_ticks)

    def update(self, trades: List[Trade], oi: float, mid: float) -> Tuple[bool, bool, str]:
        tick_vol = sum(t.size for t in trades)
        delta    = sum(t.signed_size for t in trades)

        self._vol_history.append(tick_vol)
        self._oi_history.append(oi)
        self._delta_history.append(delta)
        self._price_history.append(mid)

        if len(self._vol_history) < 10:
            return False, False, "Insufficient data"

        baseline_vol = float(np.mean(list(self._vol_history)[:-5]))
        current_vol  = float(np.mean(list(self._vol_history)[-3:]))
        vol_spike    = current_vol / (baseline_vol + 1e-9)

        ois         = list(self._oi_history)
        oi_drop_pct = (ois[-3] - ois[-1]) / (ois[-3] + 1e-9) if len(ois) >= 3 else 0.0
        cascade_starting = vol_spike > self.volume_spike_multiple and oi_drop_pct > self.oi_drop_threshold_pct

        vol_decelerating = (
            len(self._vol_history) > 5
            and list(self._vol_history)[-1] < list(self._vol_history)[-3] * 0.7
        )
        delta_reversal = (
            len(self._delta_history) >= 4
            and np.sign(sum(list(self._delta_history)[-4:-2])) != np.sign(sum(list(self._delta_history)[-2:]))
        )
        cascade_exhausting = vol_decelerating and delta_reversal

        desc = f"vol_spike={vol_spike:.1f}x oi_drop={oi_drop_pct:.3%} vol_decel={vol_decelerating} delta_rev={delta_reversal}"

        if cascade_starting:
            logger.warning("CASCADE STARTING: %s", desc)
        if cascade_exhausting:
            logger.info("CASCADE EXHAUSTING: %s", desc)

        return cascade_starting, cascade_exhausting, desc


class LiquidationFrontrunStrategy:
    """
    Two-leg execution:
      Leg 1 (taker)  — enter aggressively in cascade direction, ride forced selling.
      Leg 2 (maker)  — passive absorption bids at estimated wick level, collect reversion.
    Liquidation flow is non-informed by definition, so maker adverse selection is minimal.
    """

    def __init__(
        self,
        max_taker_size_usd: float = 50_000.0,
        max_maker_size_usd: float = 30_000.0,
        absorption_level_below_trigger_pct: float = 0.015,
        taker_target_multiple: float = 0.7,
        maker_cancel_if_no_fill_s: float = 30.0,
        min_cluster_confidence: float = 0.40,
    ) -> None:
        self.max_taker_size_usd    = max_taker_size_usd
        self.max_maker_size_usd    = max_maker_size_usd
        self.absorption_pct        = absorption_level_below_trigger_pct
        self.taker_target_multiple = taker_target_multiple
        self.maker_cancel_s        = maker_cancel_if_no_fill_s
        self.min_conf              = min_cluster_confidence

        self.cluster_mapper = LiquidationClusterMapper()
        self.cascade_det    = CascadeDetector()

        self._taker_position: Optional[dict] = None
        self._pending_maker_orders: List[dict] = []
        self._trade_log: List[dict] = []

    def update(self, book: OrderBook, trades: List[Trade], oi: float) -> dict:
        mid = book.mid
        self.cluster_mapper.update(oi, mid)
        cascade_start, cascade_exhausting, cascade_desc = self.cascade_det.update(trades, oi, mid)

        actions: dict = {
            "taker_entry": None, "taker_exit": False,
            "maker_orders": [], "maker_cancels": [],
            "cascade_state": cascade_desc,
        }

        if cascade_start and self._taker_position is None:
            self._try_taker_entry(mid, actions)

        if self._taker_position:
            self._manage_taker_exit(mid, cascade_exhausting, actions)

        self._expire_maker_orders(actions)
        return actions

    def _try_taker_entry(self, mid: float, actions: dict) -> None:
        nearby = self.cluster_mapper.get_active_clusters(mid, proximity_pct=0.03)
        if not nearby:
            return

        strongest = max(nearby, key=lambda c: c.confidence)
        if strongest.confidence < self.min_conf:
            return

        cascade_dir     = strongest.direction
        size            = self.max_taker_size_usd * strongest.confidence / mid
        cascade_distance = abs(mid - strongest.trigger_price)
        target_delta    = cascade_distance * self.taker_target_multiple

        target = mid - target_delta if cascade_dir == Side.SELL else mid + target_delta
        stop   = mid + cascade_distance * 0.2 if cascade_dir == Side.SELL else mid - cascade_distance * 0.2

        self._taker_position = {
            "side": cascade_dir, "size": size, "entry_price": mid,
            "target": target, "stop": stop, "entry_ts": time.time(),
            "cluster": strongest, "id": str(uuid.uuid4())[:8],
        }
        actions["taker_entry"] = self._taker_position.copy()
        logger.warning("TAKER ENTRY [LIQFRONT]: %s @ %.1f target=%.1f %s", cascade_dir.value, mid, target, strongest)

        if cascade_dir == Side.SELL:
            absorption_price = mid * (1 - self.absorption_pct)
            maker_order = {
                "side": Side.BUY, "price": absorption_price,
                "size": self.max_maker_size_usd / absorption_price,
                "post_only": True, "cancel_at": time.time() + self.maker_cancel_s,
                "id": str(uuid.uuid4())[:8],
            }
            self._pending_maker_orders.append(maker_order)
            actions["maker_orders"].append(maker_order)
            logger.info("ABSORPTION BID: %.1f (%.1f%% below mid)", absorption_price, self.absorption_pct * 100)

    def _manage_taker_exit(self, mid: float, cascade_exhausting: bool, actions: dict) -> None:
        pos      = self._taker_position
        at_target = (pos["side"] == Side.SELL and mid <= pos["target"]) or (pos["side"] == Side.BUY and mid >= pos["target"])
        stop_hit  = (pos["side"] == Side.SELL and mid >= pos["stop"])   or (pos["side"] == Side.BUY and mid <= pos["stop"])

        if at_target or stop_hit or cascade_exhausting:
            reason   = "TARGET" if at_target else ("STOP" if stop_hit else "CASCADE_EXHAUSTED")
            pnl_sign = -1 if pos["side"] == Side.SELL else 1
            pnl      = pnl_sign * (mid - pos["entry_price"]) * pos["size"]
            logger.info("TAKER EXIT [%s]: entry=%.1f exit=%.1f pnl=%.4f", reason, pos["entry_price"], mid, pnl)
            self._trade_log.append({"type": "taker", "pnl": pnl, "reason": reason})
            actions["taker_exit"] = True
            self._taker_position = None

    def _expire_maker_orders(self, actions: dict) -> None:
        now       = time.time()
        to_cancel = [o for o in self._pending_maker_orders if o["cancel_at"] < now]
        for o in to_cancel:
            actions["maker_cancels"].append(o["id"])
            logger.info("MAKER EXPIRED: id=%s price=%.1f", o["id"], o["price"])
        self._pending_maker_orders = [o for o in self._pending_maker_orders if o not in to_cancel]

    @property
    def total_pnl(self) -> float:
        return sum(t["pnl"] for t in self._trade_log)


def simulate_liquidation_cascade(
    n_ticks: int = 300,
    mid: float = 50_000.0,
    cascade_start_tick: int = 120,
) -> None:
    rng      = np.random.default_rng(17)
    strategy = LiquidationFrontrunStrategy()
    oi       = 1_000_000.0

    print("\n" + "="*65)
    print("LIQUIDATION FRONT-RUN SIMULATION")
    print("="*65)

    for i in range(n_ticks):
        if i < cascade_start_tick:
            mid += rng.normal(10, 5)
            oi  += rng.normal(5000, 1000)
        elif i < cascade_start_tick + 20:
            mid -= rng.normal(80, 20)
            oi  -= rng.normal(15000, 3000)
        else:
            mid += rng.normal(30, 15)

        mid    = max(45_000.0, mid)
        book   = simulate_order_book(mid=mid, rng=rng)
        trades = simulate_trade_tape(
            n_trades=rng.integers(5, 25), mid=mid, sigma_per_trade=5.0,
            informed_fraction=0.7 if cascade_start_tick <= i < cascade_start_tick + 20 else 0.1,
            rng=rng,
        )

        if cascade_start_tick <= i < cascade_start_tick + 15:
            for t in trades:
                if rng.random() < 0.4:
                    t.is_liquidation = True
                    t.side = Side.SELL

        actions = strategy.update(book, trades, oi)

        if actions["taker_entry"]:
            entry = actions["taker_entry"]
            print(f"\n[Tick {i:3d}] mid={mid:.1f}")
            print(f"  ✦ TAKER ENTRY: {entry['side'].value.upper()} @ {entry['entry_price']:.1f} | "
                  f"target={entry['target']:.1f} stop={entry['stop']:.1f}")

        if actions["taker_exit"]:
            print(f"  ✦ TAKER EXIT @ {mid:.1f} | Total PnL: {strategy.total_pnl:.4f}")

        if actions["maker_orders"]:
            for o in actions["maker_orders"]:
                print(f"  ✦ ABSORPTION {o['side'].value.upper()} @ {o['price']:.1f}")

        if i % 60 == 0:
            clusters = strategy.cluster_mapper.get_active_clusters(mid)
            print(f"[Tick {i:3d}] mid={mid:.1f} oi={oi:.0f} clusters={len(clusters)}")

    print(f"\nFinal PnL: {strategy.total_pnl:.4f}")
    print("="*65 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    simulate_liquidation_cascade()