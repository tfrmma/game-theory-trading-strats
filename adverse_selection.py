from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from init import Side, Trade, OrderBook

logger = logging.getLogger(__name__)

# Markout horizons in seconds — trade microstructure standard
MARKOUT_HORIZONS_S: Tuple[float, ...] = (5.0, 15.0, 30.0, 60.0, 300.0)


# Core data structures

@dataclass
class FillRecord:
    """
    Everything needed to compute post-fill adverse selection metrics.
    Created at fill time; mid_at_horizons populated as time elapses.
    """
    fill_id:         str
    strategy:        str
    side:            Side
    fill_price:      float
    fill_size:       float
    mid_at_fill:     float
    timestamp:       float
    is_maker:        bool

    # Populated lazily as markout windows close
    mid_at_horizons: Dict[float, float] = field(default_factory=dict)

    @property
    def direction(self) -> float:
        """+1 for buy, -1 for sell."""
        return 1.0 if self.side == Side.BUY else -1.0

    @property
    def effective_spread_bps(self) -> float:
        """
        Effective half-spread in bps = |fill - mid| / mid × 10000.
        For a maker buy below mid this is negative (we captured spread).
        """
        return (self.fill_price - self.mid_at_fill) / self.mid_at_fill * 1e4 * self.direction

    def realized_spread_bps(self, horizon_s: float) -> Optional[float]:
        """
        Realized half-spread at horizon h:
            direction × (fill_price − mid_h) / mid_fill × 10000
        Positive = we kept some of the spread. Negative = adverse selection ate it.
        """
        mid_h = self.mid_at_horizons.get(horizon_s)
        if mid_h is None:
            return None
        return self.direction * (self.fill_price - mid_h) / self.mid_at_fill * 1e4

    def price_impact_bps(self, horizon_s: float) -> Optional[float]:
        """
        Permanent price impact = effective_spread − realized_spread.
        Represents the information content of the flow that filled us.
        """
        rs = self.realized_spread_bps(horizon_s)
        if rs is None:
            return None
        return self.effective_spread_bps - rs

    def markout_pnl(self, horizon_s: float) -> Optional[float]:
        """
        PnL at horizon h:  direction × (mid_h − fill_price) × size.
        Negative = we were adversely selected and are holding a losing position.
        """
        mid_h = self.mid_at_horizons.get(horizon_s)
        if mid_h is None:
            return None
        return self.direction * (mid_h - self.fill_price) * self.fill_size


# Markout tracker

class MarkoutTracker:
    """
    Receives fills and mid updates; computes markout PnL at configured horizons.

    Usage:
        tracker.record_fill(fill)            # at fill time
        tracker.update_mid(book.mid, now)    # on every tick
        report = tracker.report()            # query at any time
    """

    def __init__(
        self,
        horizons_s: Tuple[float, ...] = MARKOUT_HORIZONS_S,
        max_open_fills: int = 5_000,
    ) -> None:
        self.horizons_s  = horizons_s
        self._open_fills: Deque[FillRecord] = deque(maxlen=max_open_fills)
        self._closed_fills: List[FillRecord] = []
        self._last_mid: float = 0.0

    def record_fill(self, fill: FillRecord) -> None:
        self._open_fills.append(fill)

    def update_mid(self, mid: float, now: float) -> None:
        self._last_mid = mid
        closed: List[FillRecord] = []

        for fill in self._open_fills:
            elapsed = now - fill.timestamp
            for h in self.horizons_s:
                if h not in fill.mid_at_horizons and elapsed >= h:
                    fill.mid_at_horizons[h] = mid

            if all(h in fill.mid_at_horizons for h in self.horizons_s):
                closed.append(fill)

        for f in closed:
            self._open_fills.remove(f)
            self._closed_fills.append(f)

    def report(self, strategy: Optional[str] = None) -> "MarkoutReport":
        fills = [f for f in self._closed_fills if strategy is None or f.strategy == strategy]
        return MarkoutReport.from_fills(fills, self.horizons_s)


@dataclass
class MarkoutReport:
    n_fills:         int
    avg_effective_spread_bps: float
    markout_pnl_by_horizon:   Dict[float, float]   # horizon → avg markout PnL per unit
    realized_spread_by_horizon: Dict[float, float]
    price_impact_by_horizon:  Dict[float, float]
    adverse_selection_rate:   float                 # fraction of fills with negative markout at 60s

    @classmethod
    def from_fills(
        cls, fills: List[FillRecord], horizons: Tuple[float, ...]
    ) -> "MarkoutReport":
        if not fills:
            return cls(0, 0.0, {}, {}, {}, 0.0)

        eff_spreads = [f.effective_spread_bps for f in fills]
        mkout_pnl:  Dict[float, List[float]] = {h: [] for h in horizons}
        real_sprd:  Dict[float, List[float]] = {h: [] for h in horizons}
        px_impact:  Dict[float, List[float]] = {h: [] for h in horizons}

        for f in fills:
            for h in horizons:
                mp = f.markout_pnl(h)
                rs = f.realized_spread_bps(h)
                pi = f.price_impact_bps(h)
                if mp is not None: mkout_pnl[h].append(mp / max(f.fill_size, 1e-9))
                if rs is not None: real_sprd[h].append(rs)
                if pi is not None: px_impact[h].append(pi)

        h60 = 60.0
        adv_rate = 0.0
        if mkout_pnl.get(h60):
            adv_rate = sum(1 for v in mkout_pnl[h60] if v < 0) / len(mkout_pnl[h60])

        return cls(
            n_fills=len(fills),
            avg_effective_spread_bps=float(np.mean(eff_spreads)),
            markout_pnl_by_horizon={h: float(np.mean(v)) for h, v in mkout_pnl.items() if v},
            realized_spread_by_horizon={h: float(np.mean(v)) for h, v in real_sprd.items() if v},
            price_impact_by_horizon={h: float(np.mean(v)) for h, v in px_impact.items() if v},
            adverse_selection_rate=adv_rate,
        )

    def print(self, label: str = "") -> None:
        header = f"ADVERSE SELECTION REPORT {('— ' + label) if label else ''}"
        print(f"\n{'='*60}")
        print(header)
        print(f"{'='*60}")
        print(f"  Fills:               {self.n_fills}")
        print(f"  Eff. spread (bps):   {self.avg_effective_spread_bps:.3f}")
        print(f"  Adverse sel. rate:   {self.adverse_selection_rate:.1%}")
        print(f"\n  {'Horizon':>8}  {'Markout PnL/unit':>18}  {'Realized sprd':>14}  {'Price impact':>13}")
        print(f"  {'-'*60}")
        for h in sorted(self.markout_pnl_by_horizon):
            mk = self.markout_pnl_by_horizon.get(h, float("nan"))
            rs = self.realized_spread_by_horizon.get(h, float("nan"))
            pi = self.price_impact_by_horizon.get(h, float("nan"))
            print(f"  {h:>7.0f}s  {mk:>18.6f}  {rs:>13.3f}bps  {pi:>12.3f}bps")
        print(f"{'='*60}\n")


# Spread decomposition (Roll model)

class RollSpreadEstimator:
    """
    Roll (1984) implied spread from serial covariance of trade price changes.
        s = 2 × sqrt(max(0, -cov(ΔP_t, ΔP_{t-1})))

    Interpretation: if trades alternate between bid and ask, successive price
    changes are negatively autocorrelated. The stronger the autocorrelation,
    the wider the implied effective spread.
    """

    def __init__(self, window: int = 200) -> None:
        self._dp: Deque[float] = deque(maxlen=window + 1)
        self._last_price: Optional[float] = None

    def update(self, trade_price: float) -> Optional[float]:
        if self._last_price is not None:
            self._dp.append(trade_price - self._last_price)
        self._last_price = trade_price
        return self.estimate

    @property
    def estimate(self) -> Optional[float]:
        dp = list(self._dp)
        if len(dp) < 30:
            return None
        arr   = np.array(dp)
        cov   = float(np.cov(arr[:-1], arr[1:])[0, 1])
        return 2.0 * np.sqrt(max(0.0, -cov))


# Amihud illiquidity

class AmihudTracker:
    """
    Amihud (2002) illiquidity ratio: |r_t| / volume_t.
    Higher = thinner book, each unit of volume causes more price impact.
    Useful for dynamic spread scaling and regime detection.
    """

    def __init__(self, window: int = 100) -> None:
        self._ratios: Deque[float] = deque(maxlen=window)
        self._prev_mid: Optional[float] = None

    def update(self, mid: float, trades: List[Trade]) -> Optional[float]:
        volume = sum(t.size for t in trades)
        if self._prev_mid is not None and volume > 0:
            ret   = abs(mid - self._prev_mid) / self._prev_mid
            ratio = ret / volume
            self._ratios.append(ratio)
        self._prev_mid = mid
        return self.current

    @property
    def current(self) -> Optional[float]:
        return float(np.mean(self._ratios)) if len(self._ratios) >= 10 else None

    def illiquidity_regime(self) -> str:
        v = self.current
        if v is None: return "UNKNOWN"
        if v < 1e-7:  return "LIQUID"
        if v < 1e-6:  return "NORMAL"
        if v < 1e-5:  return "ILLIQUID"
        return "EXTREMELY_ILLIQUID"


# Composite reporter (plugs into BacktestEngine or live runner)

class AdverseSelectionMonitor:
    """
    Combines MarkoutTracker + RollEstimator + AmihudTracker into one object.
    Call update() on every tick; create_fill_record() after each fill.
    """

    def __init__(self) -> None:
        self.markout  = MarkoutTracker()
        self.roll     = RollSpreadEstimator()
        self.amihud   = AmihudTracker()
        self._fill_seq = 0

    def update(self, book: OrderBook, trades: List[Trade]) -> dict:
        now = book.timestamp
        self.markout.update_mid(book.mid, now)
        for t in trades:
            self.roll.update(t.price)
        self.amihud.update(book.mid, trades)

        return {
            "roll_spread":  self.roll.estimate,
            "amihud":       self.amihud.current,
            "amihud_regime": self.amihud.illiquidity_regime(),
        }

    def record_fill(
        self,
        strategy: str,
        side: Side,
        fill_price: float,
        fill_size: float,
        mid_at_fill: float,
        timestamp: float,
        is_maker: bool = True,
    ) -> FillRecord:
        self._fill_seq += 1
        record = FillRecord(
            fill_id=f"fill_{self._fill_seq:06d}",
            strategy=strategy,
            side=side,
            fill_price=fill_price,
            fill_size=fill_size,
            mid_at_fill=mid_at_fill,
            timestamp=timestamp,
            is_maker=is_maker,
        )
        self.markout.record_fill(record)
        return record

    def report(self, strategy: Optional[str] = None) -> MarkoutReport:
        return self.markout.report(strategy)


# Simulation demo

def simulate_adverse_selection_metrics() -> None:
    from init import simulate_order_book, simulate_trade_tape
    import time

    monitor = AdverseSelectionMonitor()
    rng     = np.random.default_rng(77)
    mid     = 50_000.0
    now     = time.time()

    print("\n" + "="*60)
    print("ADVERSE SELECTION METRICS DEMO")
    print("="*60)

    for i in range(500):
        book   = simulate_order_book(mid=mid, rng=rng)
        trades = simulate_trade_tape(rng.integers(3, 12), mid, 5.0, 0.25 if i > 250 else 0.05, rng)

        tick_ts = now + i * 0.1
        book    = OrderBook(bids=book.bids, asks=book.asks, timestamp=tick_ts, sequence_id=i)
        state   = monitor.update(book, trades)

        # Simulate a fill every 20 ticks
        if i % 20 == 0 and trades:
            side = Side.BUY if rng.random() > 0.5 else Side.SELL
            fill_price = book.mid + (0.5 if side == Side.BUY else -0.5)
            monitor.record_fill(
                strategy="test_mm", side=side, fill_price=fill_price,
                fill_size=0.1, mid_at_fill=book.mid, timestamp=tick_ts,
            )

        if i % 100 == 0:
            print(f"[Tick {i:3d}] roll={state['roll_spread'] or 0:.4f} "
                  f"amihud={state['amihud'] or 0:.2e} regime={state['amihud_regime']}")

        mid += rng.normal(0, 6)

    monitor.report("test_mm").print("test_mm strategy")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    simulate_adverse_selection_metrics()