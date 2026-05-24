from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import numpy as np


class Side(Enum):
    BUY  = "buy"
    SELL = "sell"

    def opposite(self) -> "Side":
        return Side.SELL if self == Side.BUY else Side.BUY


class OrderType(Enum):
    LIMIT     = "limit"
    MARKET    = "market"
    POST_ONLY = "post_only"
    IOC       = "ioc"


class SignalStrength(Enum):
    WEAK    = 1
    MEDIUM  = 2
    STRONG  = 3
    EXTREME = 4


class MarketRegime(Enum):
    TRENDING    = auto()
    MEAN_REVERT = auto()
    TOXIC       = auto()
    ILLIQUID    = auto()


@dataclass
class BookLevel:
    price: float
    size: float
    order_count: int = 1

    def __post_init__(self) -> None:
        assert self.price > 0
        assert self.size >= 0


@dataclass
class OrderBook:
    bids: List[BookLevel]   # descending
    asks: List[BookLevel]   # ascending
    timestamp: float
    sequence_id: int

    def validate(self) -> bool:
        if not self.bids or not self.asks:
            return False
        if self.bids[0].price >= self.asks[0].price:
            return False
        for i in range(1, len(self.bids)):
            if self.bids[i].price >= self.bids[i - 1].price:
                return False
        for i in range(1, len(self.asks)):
            if self.asks[i].price <= self.asks[i - 1].price:
                return False
        return True

    @property
    def mid(self) -> float:
        return (self.bids[0].price + self.asks[0].price) / 2.0

    @property
    def spread(self) -> float:
        return self.asks[0].price - self.bids[0].price

    @property
    def spread_bps(self) -> float:
        return self.spread / self.mid * 1e4

    def bid_depth(self, n_levels: int = 5) -> float:
        return sum(lvl.size for lvl in self.bids[:n_levels])

    def ask_depth(self, n_levels: int = 5) -> float:
        return sum(lvl.size for lvl in self.asks[:n_levels])

    def imbalance(self, n_levels: int = 5) -> float:
        bd = self.bid_depth(n_levels)
        ad = self.ask_depth(n_levels)
        total = bd + ad
        return 0.0 if total == 0 else (bd - ad) / total


@dataclass
class Trade:
    price: float
    size: float
    side: Side
    timestamp: float
    is_liquidation: bool = False
    trade_id: Optional[str] = None

    @property
    def signed_size(self) -> float:
        return self.size if self.side == Side.BUY else -self.size


@dataclass
class FundingSnapshot:
    rate: float
    next_funding_ts: float
    open_interest: float
    mark_price: float
    index_price: float
    timestamp: float

    @property
    def annualized_rate(self) -> float:
        return self.rate * 3 * 365

    @property
    def basis_bps(self) -> float:
        return (self.mark_price - self.index_price) / self.index_price * 1e4

    @property
    def seconds_to_print(self) -> float:
        return max(0.0, self.next_funding_ts - time.time())


@dataclass
class InventoryState:
    net_position: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    spread_pnl: float = 0.0
    adverse_selection_cost: float = 0.0
    funding_pnl: float = 0.0
    trade_count: int = 0

    @property
    def unrealized_pnl(self) -> float:
        return 0.0

    def update_on_fill(
        self,
        side: Side,
        size: float,
        fill_price: float,
        mark_price: float,
        is_maker: bool,
    ) -> None:
        signed_size = size if side == Side.BUY else -size
        old_pos = self.net_position

        if old_pos == 0:
            self.avg_entry_price = fill_price
        elif (old_pos > 0 and signed_size > 0) or (old_pos < 0 and signed_size < 0):
            self.avg_entry_price = (
                (abs(old_pos) * self.avg_entry_price + size * fill_price)
                / (abs(old_pos) + size)
            )
        else:
            closed = min(abs(old_pos), size)
            pnl_sign = 1.0 if old_pos > 0 else -1.0
            self.realized_pnl += closed * pnl_sign * (fill_price - self.avg_entry_price)

        self.net_position += signed_size
        self.trade_count += 1

        if is_maker:
            self.adverse_selection_cost += abs(fill_price - mark_price) * size


@dataclass
class ExecutionOrder:
    side: Side
    price: float
    size: float
    order_type: OrderType
    client_id: str
    timestamp: float = field(default_factory=time.time)
    post_only: bool = True
    reduce_only: bool = False


@dataclass
class Signal:
    direction: Side
    strength: SignalStrength
    confidence: float
    entry_price: float
    target_price: float
    stop_price: float
    rationale: str
    timestamp: float = field(default_factory=time.time)
    invalidation_conditions: List[str] = field(default_factory=list)

    @property
    def risk_reward(self) -> float:
        reward = abs(self.target_price - self.entry_price)
        risk   = abs(self.stop_price   - self.entry_price)
        return reward / risk if risk > 0 else 0.0


def simulate_order_book(
    mid: float = 50_000.0,
    n_levels: int = 10,
    tick_size: float = 0.5,
    base_size: float = 1.0,
    size_decay: float = 0.85,
    rng: Optional[np.random.Generator] = None,
) -> OrderBook:
    if rng is None:
        rng = np.random.default_rng()

    half_spread_ticks = rng.integers(1, 3)
    best_bid = mid - half_spread_ticks * tick_size
    best_ask = mid + half_spread_ticks * tick_size

    bids, asks = [], []
    for i in range(n_levels):
        size = base_size * (size_decay ** i) * rng.lognormal(0, 0.4)
        bids.append(BookLevel(price=best_bid - i * tick_size, size=round(size, 3), order_count=rng.integers(1, 8)))
        asks.append(BookLevel(price=best_ask + i * tick_size, size=round(size, 3), order_count=rng.integers(1, 8)))

    return OrderBook(bids=bids, asks=asks, timestamp=time.time(), sequence_id=rng.integers(1_000_000, 9_999_999))


def simulate_trade_tape(
    n_trades: int,
    mid: float,
    sigma_per_trade: float = 5.0,
    informed_fraction: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> List[Trade]:
    if rng is None:
        rng = np.random.default_rng()

    trades: List[Trade] = []
    price = mid
    informed_direction = rng.choice([-1, 1])

    for i in range(n_trades):
        is_informed = rng.random() < informed_fraction
        if is_informed:
            side = Side.BUY if informed_direction > 0 else Side.SELL
            size = rng.lognormal(2.0, 0.5)
        else:
            side = rng.choice([Side.BUY, Side.SELL])
            size = rng.lognormal(0.5, 0.5)

        price += (0.5 if side == Side.BUY else -0.5) + rng.normal(0, sigma_per_trade)
        trades.append(Trade(
            price=round(price, 1),
            size=round(size, 4),
            side=side,
            timestamp=time.time() + i * 0.1,
        ))

    return trades


def compute_realized_volatility(prices: np.ndarray, window: int = 20) -> float:
    if len(prices) < 2:
        return 0.0
    return float(np.std(np.diff(np.log(prices[-window:]))))