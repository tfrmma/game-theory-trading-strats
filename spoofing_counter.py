from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from init import BookLevel, OrderBook, Trade, Side, Signal, SignalStrength, simulate_order_book, simulate_trade_tape

logger = logging.getLogger(__name__)


@dataclass
class SpoofEvent:
    side: Side
    price: float
    visible_size: float
    appeared_ts: float
    vanished_ts: Optional[float] = None
    triggered_trade: bool = False

    @property
    def duration_s(self) -> Optional[float]:
        return (self.vanished_ts - self.appeared_ts) if self.vanished_ts else None

    def __str__(self) -> str:
        return (
            f"SpoofEvent[{self.side.value}] @ {self.price:.1f} "
            f"size={self.visible_size:.2f} dur={self.duration_s:.1f}s"
            if self.duration_s else
            f"SpoofEvent[{self.side.value}] @ {self.price:.1f} size={self.visible_size:.2f} (pending)"
        )


class SpoofingDetector:
    """
    Detects spoofing via large order appearance/disappearance without trade execution.
    Spoof tell: order large enough to move apparent OFI, gone within seconds, no fill.
    """

    def __init__(
        self,
        size_anomaly_multiple: float = 5.0,
        max_spoof_duration_s: float = 8.0,
        min_spoof_size_ratio: float = 0.15,
    ) -> None:
        self.size_anomaly_multiple = size_anomaly_multiple
        self.max_spoof_duration_s  = max_spoof_duration_s
        self.min_spoof_size_ratio  = min_spoof_size_ratio

        self._prev_bids: Dict[float, float] = {}
        self._prev_asks: Dict[float, float] = {}
        self._pending_events: List[SpoofEvent] = []
        self._confirmed_events: List[SpoofEvent] = []
        self._size_history: Deque[float] = deque(maxlen=100)

    def update(self, book: OrderBook) -> List[SpoofEvent]:
        current_bids = {lvl.price: lvl.size for lvl in book.bids}
        current_asks = {lvl.price: lvl.size for lvl in book.asks}

        all_sizes = [lvl.size for lvl in book.bids + book.asks]
        self._size_history.extend(all_sizes)
        baseline_size = float(np.median(self._size_history)) if self._size_history else 1.0

        new_events = self._detect_large_appearances(current_bids, current_asks, baseline_size, book)
        confirmed  = self._detect_vanishments(current_bids, current_asks, book)

        self._prev_bids = current_bids
        self._prev_asks = current_asks

        return confirmed

    def _detect_large_appearances(
        self,
        current_bids: Dict[float, float],
        current_asks: Dict[float, float],
        baseline_size: float,
        book: OrderBook,
    ) -> List[SpoofEvent]:
        new_events: List[SpoofEvent] = []
        total_depth = book.bid_depth(10) + book.ask_depth(10)

        for price, size in current_bids.items():
            if price not in self._prev_bids and size > baseline_size * self.size_anomaly_multiple:
                if size / (total_depth + 1e-9) > self.min_spoof_size_ratio:
                    ev = SpoofEvent(side=Side.BUY, price=price, visible_size=size, appeared_ts=time.time())
                    self._pending_events.append(ev)
                    new_events.append(ev)
                    logger.debug("Large bid appeared @ %.1f size=%.2f (%.1fx baseline)", price, size, size / baseline_size)

        for price, size in current_asks.items():
            if price not in self._prev_asks and size > baseline_size * self.size_anomaly_multiple:
                if size / (total_depth + 1e-9) > self.min_spoof_size_ratio:
                    ev = SpoofEvent(side=Side.SELL, price=price, visible_size=size, appeared_ts=time.time())
                    self._pending_events.append(ev)
                    new_events.append(ev)
                    logger.debug("Large ask appeared @ %.1f size=%.2f (%.1fx baseline)", price, size, size / baseline_size)

        return new_events

    def _detect_vanishments(
        self,
        current_bids: Dict[float, float],
        current_asks: Dict[float, float],
        book: OrderBook,
    ) -> List[SpoofEvent]:
        confirmed: List[SpoofEvent] = []
        now = time.time()

        for ev in list(self._pending_events):
            current_book = current_bids if ev.side == Side.BUY else current_asks
            still_present = ev.price in current_book and current_book[ev.price] > ev.visible_size * 0.5
            age = now - ev.appeared_ts

            if not still_present and age < self.max_spoof_duration_s:
                ev.vanished_ts = now
                self._confirmed_events.append(ev)
                self._pending_events.remove(ev)
                logger.info("SPOOF DETECTED: %s", ev)
                confirmed.append(ev)
            elif age > self.max_spoof_duration_s:
                self._pending_events.remove(ev)

        return confirmed


class SpoofCounterStrategy:
    """
    Fades the direction of spoofed orders.
    Spoof bid = fake buy pressure = fade short. Spoof ask = fade long.
    Signal strength scales with spoof size and recency.
    """

    def __init__(
        self,
        min_confidence: float = 0.40,
        target_reversion_pct: float = 0.003,
        stop_loss_pct: float = 0.0015,
    ) -> None:
        self.min_confidence       = min_confidence
        self.target_reversion_pct = target_reversion_pct
        self.stop_loss_pct        = stop_loss_pct

        self._mid_history: Deque[float] = deque(maxlen=50)
        self._signal_log: List[Signal] = []

    def process_spoof_event(self, ev: SpoofEvent, mid: Optional[float] = None) -> Optional[Signal]:
        if mid is None and self._mid_history:
            mid = self._mid_history[-1]
        if mid is None:
            return None

        # Spoof bid = artificial buy pressure was fake → fade short
        # Spoof ask = artificial sell pressure was fake → fade long
        if ev.side == Side.BUY:
            direction = Side.SELL
            entry     = mid * 0.99995
            target    = mid * (1 - self.target_reversion_pct)
            stop      = mid * (1 + self.stop_loss_pct)
        else:
            direction = Side.BUY
            entry     = mid * 1.00005
            target    = mid * (1 + self.target_reversion_pct)
            stop      = mid * (1 - self.stop_loss_pct)

        duration = ev.duration_s or 1.0
        # Confidence: faster vanish + larger relative size = more likely genuine spoof
        confidence = min(1.0, (1.0 / max(duration, 0.5)) * (ev.visible_size / 10.0))

        if confidence < self.min_confidence:
            logger.debug("Spoof event confidence too low (%.2f) — skipping", confidence)
            return None

        strength = SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MEDIUM

        signal = Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_price=entry,
            target_price=target,
            stop_price=stop,
            rationale=(
                f"SPOOF FADE [{ev.side.value} spoof @ {ev.price:.1f}]: "
                f"size={ev.visible_size:.2f} dur={duration:.1f}s conf={confidence:.2f}"
            ),
            invalidation_conditions=[
                "Order re-appears at same level (not a spoof, genuine liquidity)",
                "CVD confirms the spoofed direction (real flow, not fake)",
                "Price breaks through spoof level (order was genuine support/resistance)",
            ],
        )

        self._signal_log.append(signal)
        logger.info("SPOOF COUNTER SIGNAL: %s @ %.4f", direction.value, entry)
        return signal

    def update(self, book: OrderBook) -> None:
        self._mid_history.append(book.mid)


def simulate_spoofing(n_ticks: int = 200, mid: float = 50_000.0) -> None:
    rng      = np.random.default_rng(42)
    detector = SpoofingDetector()
    strategy = SpoofCounterStrategy()

    spoof_tick    = 80
    spoof_price   = mid + 5.0
    spoof_active  = False

    print("\n" + "="*65)
    print("SPOOFING DETECTION SIMULATION")
    print("="*65)

    for i in range(n_ticks):
        book   = simulate_order_book(mid=mid, rng=rng)
        trades = simulate_trade_tape(rng.integers(3, 12), mid, 5.0, 0.1, rng)

        # Inject a large fake ask at tick 80, remove at tick 85
        if i == spoof_tick:
            book.asks.insert(0, BookLevel(price=spoof_price, size=50.0, order_count=1))
            spoof_active = True
            print(f"\n[Tick {i}] SPOOF INJECTED: ask @ {spoof_price:.1f} size=50")

        spoof_events = detector.update(book)
        strategy.update(book)

        for ev in spoof_events:
            signal = strategy.process_spoof_event(ev, mid=mid)
            print(f"\n[Tick {i}] {ev}")
            if signal:
                print(f"  → COUNTER: {signal.direction.value.upper()} @ {signal.entry_price:.2f} "
                      f"target={signal.target_price:.2f} conf={signal.confidence:.2f}")

        if i % 40 == 0:
            print(f"[Tick {i:3d}] mid={mid:.1f} pending_spoof={len(detector._pending_events)}")

        if rng.random() < 0.4:
            mid += rng.normal(0, 8)

    print(f"\nTotal confirmed spoof events: {len(detector._confirmed_events)}")
    print(f"Total counter signals: {len(strategy._signal_log)}")
    print("="*65 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    simulate_spoofing()