from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from init import (
    BookLevel, OrderBook, Trade, Side, Signal, SignalStrength,
    InventoryState, ExecutionOrder, OrderType,
    simulate_order_book, simulate_trade_tape,
    compute_realized_volatility,
)
from info_asymmetry import FlowToxicityClassifier

logger = logging.getLogger(__name__)


@dataclass
class GuerrillaQuote:
    """Single posted limit order with metadata for cancellation tracking."""
    side: Side
    price: float
    size: float
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    posted_ts: float = field(default_factory=time.time)
    reservation_price: float = 0.0
    cancel_reason: Optional[str] = None


class AvellanedaStoikovModel:
    """Avellaneda-Stoikov reservation price and optimal spread model."""

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 0.3,
        sigma: float = 0.0008,
        time_to_liquidation_s: float = 300.0,
    ) -> None:
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.T = time_to_liquidation_s

    def reservation_price(self, mid: float, inventory: float) -> float:
        skew = self.gamma * inventory * mid
        return mid - skew - (self.gamma / 2) * (self.sigma ** 2) * self.T

    def optimal_spread(self) -> float:
        return self.gamma * self.sigma**2 * self.T + (2 / self.gamma) * np.log(1 + self.gamma / self.k)

    def update_volatility(self, new_sigma: float) -> None:
        self.sigma = new_sigma


class AdaptiveGuerrillaStrategy:
    """Avellaneda-Stoikov market making with aggressive toxic flow cancellation."""

    def __init__(
        self,
        max_inventory: float = 5.0,
        base_size_usd: float = 8000.0,
        toxicity_cancel_threshold: float = 0.55,
    ) -> None:
        self.max_inventory = max_inventory
        self.base_size_usd = base_size_usd
        self.toxicity_cancel_threshold = toxicity_cancel_threshold

        self.inventory = InventoryState()
        self.as_model = AvellanedaStoikovModel()
        self.toxicity = FlowToxicityClassifier()

        self._active_quotes: Dict[str, GuerrillaQuote] = {}
        self._cancel_log: List[dict] = []
        self._mid_history: Deque[float] = deque(maxlen=100)

    def update(
        self,
        book: OrderBook,
        trades: List[Trade],
    ) -> Tuple[List[ExecutionOrder], List[str]]:
        self._mid_history.append(book.mid)

        toxicity_state = self.toxicity.update(book, trades)

        if len(self._mid_history) >= 20:
            prices = np.array(self._mid_history)
            self.as_model.update_volatility(compute_realized_volatility(prices))

        self._simulate_passive_fills(book, trades)

        actions: List[ExecutionOrder] = []
        cancels: List[str] = []

        cancels.extend(self._cancel_toxic_quotes(toxicity_state, book))

        if len(self._active_quotes) < 2:
            actions.extend(self._generate_guerrilla_quotes(book))

        return actions, cancels

    def _cancel_toxic_quotes(self, toxicity_state: dict, book: OrderBook) -> List[str]:
        cancels = []
        for qid, quote in list(self._active_quotes.items()):
            if (quote.side == Side.BUY and toxicity_state.get("directional_signal") == Side.SELL) or \
               (quote.side == Side.SELL and toxicity_state.get("directional_signal") == Side.BUY):
                if toxicity_state.get("toxicity_score", 0) > self.toxicity_cancel_threshold:
                    cancels.append(qid)
                    quote.cancel_reason = "TOXIC_FLOW"
                    self._cancel_log.append({"id": qid, "reason": "TOXIC_FLOW"})
                    del self._active_quotes[qid]
                    logger.info("GUERRILLA CANCEL: %s @ %.1f", quote.side.value, quote.price)
        return cancels

    def _generate_guerrilla_quotes(self, book: OrderBook) -> List[ExecutionOrder]:
        mid = book.mid
        inventory_skew = self.inventory.net_position

        reservation = self.as_model.reservation_price(mid, inventory_skew)
        optimal_spread = self.as_model.optimal_spread()

        orders: List[ExecutionOrder] = []

        # Bid
        bid_price = reservation - optimal_spread / 2
        bid_size = self.base_size_usd / bid_price
        if abs(inventory_skew) < self.max_inventory:
            orders.append(ExecutionOrder(
                side=Side.BUY,
                price=round(bid_price, 1),
                size=round(bid_size, 3),
                order_type=OrderType.POST_ONLY,
                client_id=f"guerrilla_bid_{int(time.time())}",
                post_only=True,
            ))

        # Ask
        ask_price = reservation + optimal_spread / 2
        ask_size = self.base_size_usd / ask_price
        if abs(inventory_skew) < self.max_inventory:
            orders.append(ExecutionOrder(
                side=Side.SELL,
                price=round(ask_price, 1),
                size=round(ask_size, 3),
                order_type=OrderType.POST_ONLY,
                client_id=f"guerrilla_ask_{int(time.time())}",
                post_only=True,
            ))

        for order in orders:
            q = GuerrillaQuote(
                side=order.side,
                price=order.price,
                size=order.size,
                reservation_price=reservation,
            )
            self._active_quotes[q.order_id] = q

        return orders

    def _simulate_passive_fills(self, book: OrderBook, trades: List[Trade]) -> None:
        for t in trades:
            for qid, quote in list(self._active_quotes.items()):
                if abs(t.price - quote.price) < 0.01 and t.side == quote.side.opposite():
                    self.inventory.update_on_fill(
                        side=quote.side,
                        size=quote.size,
                        fill_price=t.price,
                        mark_price=book.mid,
                        is_maker=True,
                    )
                    del self._active_quotes[qid]
                    break

    @property
    def cancel_rate(self) -> float:
        total = len(self._cancel_log) + len(self._active_quotes)
        return len(self._cancel_log) / max(1, total)


def simulate_adaptive_guerrilla(n_ticks: int = 400, mid: float = 50_000.0) -> None:
    rng = np.random.default_rng(123)
    strategy = AdaptiveGuerrillaStrategy()

    for i in range(n_ticks):
        book = simulate_order_book(mid=mid, rng=rng)
        trades = simulate_trade_tape(
            n_trades=rng.integers(4, 18),
            mid=mid,
            informed_fraction=0.45 if i > 200 else 0.08,
            rng=rng,
        )

        new_orders, cancels = strategy.update(book, trades)

        if i % 80 == 0:
            print(
                f"[TICK {i:3d}] mid={mid:.1f} | inv={strategy.inventory.net_position:.2f} | "
                f"cancel_rate={strategy.cancel_rate:.1%} | toxicity={strategy.toxicity.update(book, trades)['toxicity_score']:.2f}"
            )

        if rng.random() < 0.4:
            mid += rng.normal(0, 8)

    print(f"Final inventory: {strategy.inventory.net_position:.2f}")
    print(f"Total cancels: {len(strategy._cancel_log)}")
    print(f"Realized PnL: {strategy.inventory.realized_pnl:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    simulate_adaptive_guerrilla()