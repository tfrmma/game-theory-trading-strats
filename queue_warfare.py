from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import poisson

from init import BookLevel, OrderBook, Trade, Side, simulate_order_book, simulate_trade_tape

logger = logging.getLogger(__name__)


@dataclass
class QueuePosition:
    """
    Our estimated FIFO position at a price level.
    Without L3, we estimate by tracking depth at submit time vs consumed since.
    """
    price: float
    side: Side
    our_size: float
    submitted_queue_depth: float
    consumed_since_join: float = 0.0
    submit_ts: float = field(default_factory=time.time)
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def estimated_ahead_of_us(self) -> float:
        return max(0.0, self.submitted_queue_depth - self.consumed_since_join)

    @property
    def fill_probability(self) -> float:
        if self.submitted_queue_depth == 0:
            return 0.0
        return min(1.0, self.consumed_since_join / self.submitted_queue_depth)

    @property
    def queue_age_s(self) -> float:
        return time.time() - self.submit_ts


class IcebergDetector:
    """
    Detects iceberg orders via level replenishment after trades.
    Iceberg ahead = fill probability model underestimates true queue depth.
    """

    def __init__(self, replenish_threshold: float = 5.0) -> None:
        self.replenish_threshold = replenish_threshold
        self._prev_bid_sizes: Dict[float, float] = {}
        self._prev_ask_sizes: Dict[float, float] = {}
        self._replenishment_by_level: Dict[float, float] = {}
        self.confirmed_icebergs: Dict[float, float] = {}

    def update(self, book: OrderBook, trades: List[Trade]) -> Dict[float, float]:
        current_bid = {lvl.price: lvl.size for lvl in book.bids}
        current_ask = {lvl.price: lvl.size for lvl in book.asks}

        trade_by_price: Dict[float, float] = {}
        for t in trades:
            trade_by_price[t.price] = trade_by_price.get(t.price, 0) + t.size

        for price, prev_size in self._prev_bid_sizes.items():
            if price not in current_bid:
                continue
            replenish = current_bid[price] - (prev_size - trade_by_price.get(price, 0.0))
            if replenish > 0:
                self._replenishment_by_level[price] = self._replenishment_by_level.get(price, 0) + replenish

        for price, total_replenish in self._replenishment_by_level.items():
            if total_replenish > self.replenish_threshold:
                if price not in self.confirmed_icebergs:
                    logger.info("ICEBERG DETECTED @ %.1f est_hidden=%.2f", price, total_replenish)
                self.confirmed_icebergs[price] = total_replenish

        self._prev_bid_sizes = current_bid
        self._prev_ask_sizes = current_ask
        return self.confirmed_icebergs



class LevelCancellationTracker:
    """
    Estimates the per-level cancellation ratio alpha = cancelled_vol / (cancelled_vol + traded_vol).

    On each tick we compute the size that disappeared from a level without a matching trade —
    that delta is pure cancellation. Rolling alpha tells us how fast the queue drains ahead of us
    from cancellations alone, which the naive Poisson fill model completely misses.

    Adjusted effective queue ahead:
        effective_ahead = queue_ahead * (1 - alpha)

    Higher alpha → queue drains faster than trades suggest → fill probability is higher
    than the raw Poisson estimate, and the Winner's Curse risk is lower.
    """

    def __init__(self, window: int = 50, min_obs: int = 10) -> None:
        self.window   = window
        self.min_obs  = min_obs
        self._cancel_vol: Dict[float, Deque[float]] = {}
        self._trade_vol:  Dict[float, Deque[float]] = {}
        self._prev_sizes: Dict[float, float] = {}

    def update(self, book: OrderBook, trades: List[Trade]) -> None:
        current_sizes: Dict[float, float] = {}
        for lvl in book.bids + book.asks:
            current_sizes[lvl.price] = lvl.size

        trade_by_price: Dict[float, float] = {}
        for t in trades:
            trade_by_price[t.price] = trade_by_price.get(t.price, 0.0) + t.size

        for price, prev_size in self._prev_sizes.items():
            if price not in current_sizes:
                # Level vanished entirely — all remaining size was cancelled
                cancelled = prev_size - trade_by_price.get(price, 0.0)
            else:
                size_drop = prev_size - current_sizes[price]
                traded    = trade_by_price.get(price, 0.0)
                cancelled = max(0.0, size_drop - traded)

            traded_here = trade_by_price.get(price, 0.0)

            if price not in self._cancel_vol:
                self._cancel_vol[price] = deque(maxlen=self.window)
                self._trade_vol[price]  = deque(maxlen=self.window)

            self._cancel_vol[price].append(cancelled)
            self._trade_vol[price].append(traded_here)

        self._prev_sizes = {p: s for p, s in current_sizes.items()}

    def alpha(self, price: float) -> float:
        """
        Cancellation ratio at price level ∈ [0, 1).
        Returns 0.0 if insufficient observations.
        """
        c_hist = self._cancel_vol.get(price)
        t_hist = self._trade_vol.get(price)
        if not c_hist or len(c_hist) < self.min_obs:
            return 0.0
        total_cancel = sum(c_hist)
        total_trade  = sum(t_hist)
        denom        = total_cancel + total_trade
        return total_cancel / denom if denom > 0 else 0.0

    def effective_queue_ahead(self, price: float, raw_queue_ahead: float) -> float:
        """Scale raw queue depth by (1 - alpha) to account for cancellations draining it."""
        return raw_queue_ahead * (1.0 - self.alpha(price))


class QueueFillModel:
    """
    P(fill within horizon) using Poisson market order arrivals.
    λ estimated from rolling trade rate; μ_v from rolling trade size.
    """

    def __init__(self, window_s: float = 30.0) -> None:
        self.window_s = window_s
        self._trade_times: Deque[float] = deque()
        self._trade_sizes: Deque[float] = deque()

    def update(self, trades: List[Trade]) -> None:
        now = time.time()
        for t in trades:
            self._trade_times.append(now)
            self._trade_sizes.append(t.size)

        cutoff = now - self.window_s
        while self._trade_times and self._trade_times[0] < cutoff:
            self._trade_times.popleft()
            self._trade_sizes.popleft()

    @property
    def arrival_rate(self) -> float:
        if len(self._trade_times) < 2:
            return 1.0
        elapsed = self._trade_times[-1] - self._trade_times[0]
        return len(self._trade_times) / max(elapsed, 0.001)

    @property
    def mean_trade_size(self) -> float:
        return float(np.mean(self._trade_sizes)) if self._trade_sizes else 1.0

    def fill_probability(self, queue_ahead: float, horizon_s: float = 10.0) -> float:
        λ, μv = self.arrival_rate, self.mean_trade_size
        if μv == 0 or λ == 0:
            return 0.0
        k = queue_ahead / μv
        return float(np.clip(1.0 - poisson.cdf(k - 1, mu=λ * horizon_s), 0.0, 1.0))

    def optimal_cancel_threshold(self, lambda_kyle: float, fill_spread: float) -> float:
        if lambda_kyle == 0:
            return float("inf")
        return fill_spread / lambda_kyle


class QueueWarfareStrategy:
    """
    Queue position management: join when priority is good, cancel fast when OFI turns.
    Iceberg detection adjusts fill probability estimates after order placement.
    """

    def __init__(
        self,
        max_queue_fraction_to_join: float = 0.30,
        ofi_cancel_threshold: float = 200.0,
        fill_horizon_s: float = 10.0,
        min_fill_probability: float = 0.15,
    ) -> None:
        self.max_queue_fraction_to_join = max_queue_fraction_to_join
        self.ofi_cancel_threshold       = ofi_cancel_threshold
        self.fill_horizon_s             = fill_horizon_s
        self.min_fill_probability       = min_fill_probability

        self._active_positions: Dict[str, QueuePosition] = {}
        self.fill_model    = QueueFillModel()
        self.iceberg_det   = IcebergDetector()
        self.cancel_tracker = LevelCancellationTracker()
        self._ofi_history: Deque[float] = deque(maxlen=20)
        self._cancel_log: List[dict] = []

    def should_join_queue(self, book: OrderBook, side: Side, price: float) -> Tuple[bool, str]:
        levels    = book.bids if side == Side.BUY else book.asks
        level_map = {lvl.price: lvl for lvl in levels}

        if price not in level_map:
            return False, "Level not in book"

        lvl           = level_map[price]
        total_depth   = book.bid_depth(10) if side == Side.BUY else book.ask_depth(10)
        queue_fraction = lvl.size / total_depth

        if queue_fraction > self.max_queue_fraction_to_join:
            return False, f"Queue too full ({queue_fraction:.2f})"

        alpha           = self.cancel_tracker.alpha(price)
        eff_queue_ahead = self.cancel_tracker.effective_queue_ahead(price, lvl.size)
        p_fill          = self.fill_model.fill_probability(eff_queue_ahead, self.fill_horizon_s)
        if p_fill < self.min_fill_probability:
            return False, f"Fill prob too low ({p_fill:.3f}, alpha={alpha:.2f})"

        if price in self.iceberg_det.confirmed_icebergs:
            hidden     = self.iceberg_det.confirmed_icebergs[price]
            eff_hidden = self.cancel_tracker.effective_queue_ahead(price, lvl.size + hidden)
            p_fill_adj = self.fill_model.fill_probability(eff_hidden, self.fill_horizon_s)
            if p_fill_adj < self.min_fill_probability:
                return False, f"Iceberg; adj fill prob too low ({p_fill_adj:.3f})"

        return True, f"p_fill={p_fill:.3f} alpha={alpha:.2f} queue_fraction={queue_fraction:.3f}"

    def update(self, book: OrderBook, trades: List[Trade]) -> List[str]:
        self.fill_model.update(trades)
        self.iceberg_det.update(book, trades)
        self.cancel_tracker.update(book, trades)

        ofi = sum(t.signed_size for t in trades)
        self._ofi_history.append(ofi)
        recent_ofi = float(np.sum(self._ofi_history))

        for pos in self._active_positions.values():
            traded_at_level = sum(t.size for t in trades if abs(t.price - pos.price) < 0.01)
            # Cancellations ahead of us also advance our queue position.
            # Scale traded consumption up by 1/(1-alpha) to approximate total drain rate.
            alpha = self.cancel_tracker.alpha(pos.price)
            drain_multiplier = 1.0 / max(1.0 - alpha, 0.01)
            pos.consumed_since_join += traded_at_level * drain_multiplier

        cancel_ids: List[str] = []
        for order_id, pos in list(self._active_positions.items()):
            should_cancel, reason = self._should_cancel(pos, recent_ofi)
            if should_cancel:
                cancel_ids.append(order_id)
                del self._active_positions[order_id]
                self._cancel_log.append({"ts": time.time(), "order_id": order_id, "price": pos.price, "reason": reason})
                logger.info("CANCEL order=%s price=%.1f reason=%s", order_id, pos.price, reason)

        return cancel_ids

    def _should_cancel(self, pos: QueuePosition, recent_ofi: float) -> Tuple[bool, str]:
        if pos.side == Side.BUY and recent_ofi < -self.ofi_cancel_threshold:
            return True, f"OFI={recent_ofi:.1f} against BID"
        if pos.side == Side.SELL and recent_ofi > self.ofi_cancel_threshold:
            return True, f"OFI={recent_ofi:.1f} against ASK"

        if pos.price in self.iceberg_det.confirmed_icebergs:
            hidden    = self.iceberg_det.confirmed_icebergs[pos.price]
            p_fill_adj = self.fill_model.fill_probability(pos.estimated_ahead_of_us + hidden, 5.0)
            if p_fill_adj < 0.05:
                return True, f"Iceberg ahead, adj p_fill={p_fill_adj:.3f}"

        if pos.queue_age_s > 30.0 and pos.fill_probability < 0.05:
            return True, "Stale, low fill probability"

        return False, ""

    def register_order(self, pos: QueuePosition) -> None:
        self._active_positions[pos.order_id] = pos

    @property
    def cancel_rate(self) -> float:
        recent_cutoff = time.time() - 60.0
        recent = sum(1 for c in self._cancel_log if c["ts"] > recent_cutoff)
        return recent / max(1, len(self._cancel_log))


def simulate_queue_warfare(n_ticks: int = 200, mid: float = 50_000.0) -> None:
    rng      = np.random.default_rng(13)
    strategy = QueueWarfareStrategy()

    our_order = QueuePosition(
        price=mid - 2.0, side=Side.BUY, our_size=1.0, submitted_queue_depth=15.0
    )
    strategy.register_order(our_order)

    print("\n" + "="*65)
    print("QUEUE WARFARE — FIFO OPTIONALITY")
    print("="*65)
    print(f"\nRegistered BID @ {our_order.price:.1f} | Queue ahead: {our_order.submitted_queue_depth}")

    for i in range(n_ticks):
        book         = simulate_order_book(mid=mid, rng=rng)
        informed_frac = 0.50 if i > 120 else 0.05
        trades       = simulate_trade_tape(rng.integers(3, 10), mid, 5.0, informed_frac, rng)

        ask_price    = book.asks[0].price
        can_join, reason = strategy.should_join_queue(book, Side.SELL, ask_price)
        cancel_ids   = strategy.update(book, trades)

        if cancel_ids:
            print(f"\n[Tick {i}] CANCEL: {cancel_ids}")
            for log in strategy._cancel_log[-len(cancel_ids):]:
                print(f"  {log['reason']}")

        if i == 120:
            print(f"\n[Tick {i}] *** INFORMED SELL FLOW (50%) ***")

        if i % 40 == 0:
            alpha    = strategy.cancel_tracker.alpha(our_order.price)
            eff_ahead = strategy.cancel_tracker.effective_queue_ahead(
                our_order.price, max(0, our_order.estimated_ahead_of_us)
            )
            p_fill   = strategy.fill_model.fill_probability(eff_ahead, 10.0)
            ofi      = sum(t.signed_size for t in trades)
            print(
                f"[Tick {i:3d}] mid={mid:.1f} | p_fill={p_fill:.3f} | alpha={alpha:.2f} | "
                f"consumed={our_order.consumed_since_join:.2f} | eff_ahead={eff_ahead:.2f} | "
                f"ofi={ofi:.2f} | join_ask={can_join} ({reason[:25]})"
            )

        if rng.random() < 0.3:
            mid += rng.normal(0, 5.0)

    icebergs = strategy.iceberg_det.confirmed_icebergs
    print(f"\nIcebergs detected: {len(icebergs)}")
    for price, hidden in icebergs.items():
        print(f"  {price:.1f}: est hidden = {hidden:.2f}")
    print(f"Total cancels: {len(strategy._cancel_log)}")
    print("="*65 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    simulate_queue_warfare()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    simulate_queue_warfare()
