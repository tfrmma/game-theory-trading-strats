from __future__ import annotations

import csv
import heapq
import inspect
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from init import (
    BookLevel, ExecutionOrder, InventoryState, OrderBook,
    OrderType, Side, Trade, simulate_order_book, simulate_trade_tape,
)

logger = logging.getLogger(__name__)

class HistoricalTick(NamedTuple):
    timestamp: float
    book:      OrderBook
    trades:    List[Trade]

@dataclass
class SimulatedFill:
    strategy:      str
    order_id:      str
    side:          Side
    fill_price:    float
    fill_size:     float
    mid_at_fill:   float
    timestamp:     float
    is_maker:      bool  = True
    queue_model:   str   = "simple"   # "simple" | "fifo"
    qty_in_front_at_fill: float = 0.0 # how much queue was ahead when fill triggered

    @property
    def signed_fill_size(self) -> float:
        return self.fill_size if self.side == Side.BUY else -self.fill_size

@dataclass
class StrategyMetrics:
    realized_pnl:           float = 0.0
    spread_pnl:             float = 0.0
    adverse_selection_cost: float = 0.0
    fill_count:             int   = 0
    order_count:            int   = 0
    cancel_count:           int   = 0
    signal_count:           int   = 0
    # Pro metrics
    total_queue_advancement: float = 0.0   # cumulative queue advance from cancels
    avg_order_latency_us:    float = 0.0
    latency_samples:         int   = 0

    @property
    def avg_pnl_per_fill(self) -> float:
        return self.realized_pnl / max(1, self.fill_count)

    @property
    def fill_rate(self) -> float:
        return self.fill_count / max(1, self.order_count)

    def record_latency(self, latency_us: float) -> None:
        n = self.latency_samples
        self.avg_order_latency_us = (self.avg_order_latency_us * n + latency_us) / (n + 1)
        self.latency_samples += 1

@dataclass
class BacktestReport:
    per_strategy: Dict[str, StrategyMetrics]
    equity_curve: List[Tuple[float, float]]
    fills:        List[SimulatedFill]
    total_ticks:  int
    duration_s:   float
    coin:         str
    mode:         str = "simple"   # "simple" | "pro"

    @property
    def total_pnl(self) -> float:
        return sum(m.realized_pnl for m in self.per_strategy.values())

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        pnls = np.array([p for _, p in self.equity_curve])
        return float(np.max(np.maximum.accumulate(pnls) - pnls))

    @property
    def sharpe(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        rets = np.diff([p for _, p in self.equity_curve])
        return float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(len(rets)))

    def print_summary(self) -> None:
        print(f"\n{'='*75}")
        print(f"BACKTEST [{self.mode.upper()}]  |  {self.coin}  |  "
              f"{self.total_ticks} ticks  |  {self.duration_s:.2f}s")
        print(f"{'='*75}")
        hdr = f"{'Strategy':<22} {'PnL':>10} {'Fills':>6} {'Fill%':>6} {'AdvSel':>10}"
        if self.mode == "pro":
            hdr += f"  {'QueueAdv':>10}  {'Lat(μs)':>8}"
        print(hdr)
        print(f"{'-'*75}")
        for name, m in sorted(self.per_strategy.items()):
            row = (f"{name:<22} {m.realized_pnl:>10.4f} {m.fill_count:>6d} "
                   f"{m.fill_rate:>6.1%} {m.adverse_selection_cost:>10.4f}")
            if self.mode == "pro":
                row += f"  {m.total_queue_advancement:>10.2f}  {m.avg_order_latency_us:>8.1f}"
            print(row)
        print(f"{'-'*75}")
        print(f"{'TOTAL':<22} {self.total_pnl:>10.4f}")
        print(f"\nSharpe: {self.sharpe:.3f}  |  Max Drawdown: {self.max_drawdown:.4f}")
        print(f"{'='*75}\n")

@dataclass
class LatencyConfig:
    feed_latency_us:  float = 100.0
    order_latency_us: float = 500.0
    jitter_pct:       float = 0.20

    def sample_feed_delay_s(self, rng: np.random.Generator) -> float:
        sigma = np.log1p(self.jitter_pct ** 2) ** 0.5
        mu    = np.log(self.feed_latency_us / 1e6) - sigma ** 2 / 2
        return float(rng.lognormal(mu, sigma))

    def sample_order_delay_s(self, rng: np.random.Generator) -> float:
        sigma = np.log1p(self.jitter_pct ** 2) ** 0.5
        mu    = np.log(self.order_latency_us / 1e6) - sigma ** 2 / 2
        return float(rng.lognormal(mu, sigma))

class ReduceRatioCancelModel:
    """
    Fixed fraction of cancellations occur in front of our queue position.
    Literature: 0.10–0.40 depending on venue and proximity to BBO.
    Crypto perps near BBO → ~0.20 is a reasonable default.
    """

    def __init__(self, ratio: float = 0.20) -> None:
        self.ratio = ratio

    def cancel_ratio(
        self,
        qty_in_front: float,
        total_depth:  float,
        price:        float,
        side:         Side,
    ) -> float:
        return self.ratio

class ProbQueueCancelModel:
    """
    Pro-rata model: cancellation probability in front scales with our
    position in the queue relative to total depth.

        ratio = qty_in_front / max(total_depth, 1)

    Intuition: if we're near the front, most of the book is behind us,
    so most cancels are behind us too. If we're near the back, the
    opposite — most cancels land in front of us.
    Clipped to [min_ratio, max_ratio] to avoid degenerate extremes.
    """

    def __init__(self, min_ratio: float = 0.05, max_ratio: float = 0.70) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def cancel_ratio(
        self,
        qty_in_front: float,
        total_depth:  float,
        price:        float,
        side:         Side,
    ) -> float:
        if total_depth < 1e-9:
            return self.min_ratio
        raw = qty_in_front / total_depth
        return float(np.clip(raw, self.min_ratio, self.max_ratio))

@dataclass
class FIFOQueueState:
    order_id:      str
    strategy:      str
    side:          Side
    price:         float
    remaining:     float          # unfilled size
    qty_in_front:  float          # decreases toward 0 as queue drains
    submitted_ts:  float          # wall-clock time strategy decided
    arrival_ts:    float          # when order actually lands at exchange
    iceberg_hidden: float = 0.0   # extra hidden depth detected ahead of us

    @property
    def effective_queue_ahead(self) -> float:
        return self.qty_in_front + self.iceberg_hidden

    def process_trade(self, vol_at_level: float) -> float:
        """
        Apply aggressive trade volume at our price level.
        Returns the fill amount for our order (0 if queue still ahead of us).
        """
        if self.effective_queue_ahead > 0:
            consumed = min(vol_at_level, self.effective_queue_ahead)
            # Drain iceberg hidden first (it's conceptually in front)
            iceberg_drain     = min(consumed, self.iceberg_hidden)
            self.iceberg_hidden = max(0.0, self.iceberg_hidden - iceberg_drain)
            remaining_drain   = consumed - iceberg_drain
            self.qty_in_front = max(0.0, self.qty_in_front - remaining_drain)
            vol_at_level     -= consumed

        if vol_at_level <= 0 or self.effective_queue_ahead > 0:
            return 0.0

        fill           = min(vol_at_level, self.remaining)
        self.remaining = max(0.0, self.remaining - fill)
        return fill

    def process_cancellation(self, cancel_vol: float, ratio: float) -> float:
        """
        Advance our queue position by the estimated in-front fraction.
        Returns actual advancement applied.
        """
        advancement   = cancel_vol * ratio
        old_front     = self.qty_in_front
        self.qty_in_front = max(0.0, self.qty_in_front - advancement)
        return old_front - self.qty_in_front   # actual delta

class FIFOQueueSimulator:

    def __init__(
        self,
        cancel_model:          Optional[object] = None,
        iceberg_replenish_thr: float = 3.0,
    ) -> None:
        self.cancel_model          = cancel_model or ReduceRatioCancelModel()
        self.iceberg_replenish_thr = iceberg_replenish_thr
        self._active:   Dict[str, FIFOQueueState] = {}
        self._prev_depth: Dict[float, float] = {}   # price → size at previous tick

    def register(self, state: FIFOQueueState) -> None:
        self._active[state.order_id] = state

    def cancel(self, order_id: str) -> bool:
        return self._active.pop(order_id, None) is not None

    def process_tick(
        self,
        book:       OrderBook,
        trades:     List[Trade],
    ) -> List[Tuple[FIFOQueueState, float]]:
        trade_by_price  = self._aggregate_trades(trades)
        cancel_by_price = self._infer_cancels(book, trade_by_price)
        current_depth   = self._build_depth_map(book)
        iceberg_by_price = self._detect_icebergs(book, trade_by_price, current_depth)

        fills: List[Tuple[FIFOQueueState, float]] = []

        for state in list(self._active.values()):
            # Update iceberg hidden depth for this level
            if state.price in iceberg_by_price:
                state.iceberg_hidden = max(
                    state.iceberg_hidden, iceberg_by_price[state.price]
                )

            # Apply cancellations (advance queue)
            cancel_vol = cancel_by_price.get(state.price, 0.0)
            if cancel_vol > 0:
                total_depth = current_depth.get(state.price, state.qty_in_front)
                ratio = self.cancel_model.cancel_ratio(
                    state.qty_in_front, total_depth, state.price, state.side
                )
                state.process_cancellation(cancel_vol, ratio)

            # Apply trades (fill if at front)
            trade_vol = trade_by_price.get(state.price, 0.0)
            if trade_vol > 0:
                fill = state.process_trade(trade_vol)
                if fill > 0.0:
                    fills.append((state, fill))
                    if state.remaining <= 1e-9:
                        del self._active[state.order_id]

        self._prev_depth = current_depth
        return fills

    @staticmethod
    def _aggregate_trades(trades: List[Trade]) -> Dict[float, float]:
        vol: Dict[float, float] = {}
        for t in trades:
            vol[t.price] = vol.get(t.price, 0.0) + t.size
        return vol

    def _infer_cancels(
        self,
        book:             OrderBook,
        trade_by_price:   Dict[float, float],
    ) -> Dict[float, float]:
        cancels: Dict[float, float] = {}
        current = self._build_depth_map(book)
        for price, prev_size in self._prev_depth.items():
            curr_size  = current.get(price, 0.0)
            trade_vol  = trade_by_price.get(price, 0.0)
            cancel_vol = max(0.0, prev_size - curr_size - trade_vol)
            if cancel_vol > 1e-9:
                cancels[price] = cancel_vol
        return cancels

    def _detect_icebergs(
        self,
        book:           OrderBook,
        trade_by_price: Dict[float, float],
        current_depth:  Dict[float, float],
    ) -> Dict[float, float]:
        icebergs: Dict[float, float] = {}
        for price, prev_size in self._prev_depth.items():
            curr_size  = current_depth.get(price, 0.0)
            trade_vol  = trade_by_price.get(price, 0.0)
            replenish  = curr_size - (prev_size - trade_vol)
            if replenish > self.iceberg_replenish_thr:
                icebergs[price] = replenish
                logger.debug("ICEBERG @ %.1f replenish=%.2f", price, replenish)
        return icebergs

    @staticmethod
    def _build_depth_map(book: OrderBook) -> Dict[float, float]:
        dm: Dict[float, float] = {}
        for lvl in book.bids + book.asks:
            dm[lvl.price] = lvl.size
        return dm

    def active_count(self) -> int:
        return len(self._active)

@dataclass
class _PendingOrder:
    """Heap element: order in flight toward the exchange."""
    arrival_ts:    float
    order:         ExecutionOrder
    strategy:      str
    decision_book: OrderBook       # stale book the strategy saw

    def __lt__(self, other: "_PendingOrder") -> bool:
        return self.arrival_ts < other.arrival_ts

class LatencySimulator:
    """Feed staleness + order flight time. Strategy sees stale book; orders land with delay."""

    def __init__(self, config: LatencyConfig, rng: np.random.Generator) -> None:
        self._cfg  = config
        self._rng  = rng
        self._heap: List[_PendingOrder] = []
        # Rolling buffer of (timestamp, book) for stale-feed lookup
        self._book_history: Deque[Tuple[float, OrderBook]] = deque(maxlen=500)

    def record_book(self, ts: float, book: OrderBook) -> None:
        self._book_history.append((ts, book))

    def stale_book(self, now: float) -> Optional[OrderBook]:
        target = now - self._cfg.sample_feed_delay_s(self._rng)
        for ts, book in reversed(self._book_history):
            if ts <= target:
                return book
        # Fallback during warm-up: return oldest available
        return self._book_history[0][1] if self._book_history else None

    def schedule(
        self,
        order:    ExecutionOrder,
        strategy: str,
        now:      float,
        decision_book: OrderBook,
    ) -> float:
        """Queue an order; returns the sampled arrival timestamp."""
        delay      = self._cfg.sample_order_delay_s(self._rng)
        arrival_ts = now + delay
        heapq.heappush(self._heap, _PendingOrder(arrival_ts, order, strategy, decision_book))
        return arrival_ts

    def poll_arrived(self, now: float) -> List[_PendingOrder]:
        arrived: List[_PendingOrder] = []
        while self._heap and self._heap[0].arrival_ts <= now:
            arrived.append(heapq.heappop(self._heap))
        return arrived

    def in_flight_count(self) -> int:
        return len(self._heap)

class PassiveFillSimulator:

    def __init__(self, queue_depth_lookback: int = 20) -> None:
        self._depth_hist: Deque[float] = deque(maxlen=queue_depth_lookback)

    def simulate(
        self,
        order:  ExecutionOrder,
        trades: List[Trade],
        book:   OrderBook,
    ) -> Optional[SimulatedFill]:
        matching = [
            t for t in trades if (
                (order.side == Side.BUY  and t.side == Side.SELL and t.price <= order.price) or
                (order.side == Side.SELL and t.side == Side.BUY  and t.price >= order.price)
            )
        ]
        if not matching:
            return None

        levels = book.bids if order.side == Side.BUY else book.asks
        depth  = next((lvl.size for lvl in levels if abs(lvl.price - order.price) < 0.01), 1.0)
        self._depth_hist.append(depth)
        avg_depth  = float(np.mean(self._depth_hist))
        fill_size  = min(order.size, sum(t.size for t in matching) * order.size / max(avg_depth, order.size))

        if fill_size < 1e-8:
            return None

        return SimulatedFill(
            strategy=order.client_id.split("_")[0],
            order_id=order.client_id,
            side=order.side,
            fill_price=float(np.mean([t.price for t in matching])),
            fill_size=fill_size,
            mid_at_fill=book.mid,
            timestamp=book.timestamp,
            queue_model="simple",
        )

class TickLoader:
    """
    Loads historical ticks from CSV or generates synthetic replay data.

    CSV expected columns:
        timestamp, bid1_px, bid1_sz, [bid2_px, bid2_sz, ...],
        ask1_px,   ask1_sz, [ask2_px, ask2_sz, ...],
        trade_px,  trade_sz, trade_side  (B / S)
    """

    @staticmethod
    def from_csv(path: str | Path, n_book_levels: int = 5) -> List[HistoricalTick]:
        path  = Path(path)
        ticks: Dict[float, Tuple[Optional[OrderBook], List[Trade]]] = {}

        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                ts = float(row["timestamp"])
                if ts not in ticks:
                    ticks[ts] = (TickLoader._parse_book(row, n_book_levels, ts), [])
                trade = TickLoader._parse_trade(row, ts)
                if trade:
                    ticks[ts][1].append(trade)

        result = [
            HistoricalTick(timestamp=ts, book=b, trades=tr)
            for ts in sorted(ticks)
            for b, tr in [ticks[ts]] if b
        ]
        logger.info("Loaded %d ticks from %s", len(result), path)
        return result

    @staticmethod
    def _parse_book(row: dict, n: int, ts: float) -> Optional[OrderBook]:
        try:
            bids = [BookLevel(float(row[f"bid{i}_px"]), float(row[f"bid{i}_sz"])) for i in range(1, n+1)
                    if float(row.get(f"bid{i}_px", 0) or 0) > 0]
            asks = [BookLevel(float(row[f"ask{i}_px"]), float(row[f"ask{i}_sz"])) for i in range(1, n+1)
                    if float(row.get(f"ask{i}_px", 0) or 0) > 0]
            return OrderBook(bids=bids, asks=asks, timestamp=ts, sequence_id=0) if bids and asks else None
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _parse_trade(row: dict, ts: float) -> Optional[Trade]:
        try:
            px = float(row.get("trade_px") or 0)
            sz = float(row.get("trade_sz") or 0)
            if px <= 0 or sz <= 0:
                return None
            return Trade(
                price=px, size=sz,
                side=Side.BUY if str(row.get("trade_side", "B")).upper() == "B" else Side.SELL,
                timestamp=ts,
            )
        except (KeyError, ValueError):
            return None

    @staticmethod
    def synthetic(
        n_ticks:      int   = 1_000,
        mid0:         float = 50_000.0,
        informed_ramp: bool = True,
        tick_interval_s: float = 0.05,
        seed:         int   = 42,
    ) -> List[HistoricalTick]:
        rng = np.random.default_rng(seed)
        mid = mid0
        ts  = time.time()
        out: List[HistoricalTick] = []

        for i in range(n_ticks):
            informed = 0.40 if (informed_ramp and i > n_ticks // 2) else 0.08
            book     = simulate_order_book(mid=mid, rng=rng)
            trades   = simulate_trade_tape(int(rng.integers(3, 15)), mid, 5.0, informed, rng)
            out.append(HistoricalTick(timestamp=ts + i * tick_interval_s, book=book, trades=trades))
            if rng.random() < 0.5:
                mid += rng.normal(0, 8)

        return out

def _call_strategy(strat, book: OrderBook, trades: List[Trade], oi: float):
    n = len(inspect.signature(strat.update).parameters)
    return strat.update(book, trades, oi) if n >= 3 else strat.update(book, trades)

def _extract_orders(result) -> List[ExecutionOrder]:
    if result is None:
        return []
    if isinstance(result, list):
        return [r for r in result if isinstance(r, ExecutionOrder)]
    if isinstance(result, tuple):
        orders = []
        for item in result:
            if isinstance(item, list):
                orders.extend(r for r in item if isinstance(r, ExecutionOrder))
        return orders
    return []

class BacktestEngine:
    """Simple tick-by-tick engine with conservative passive fill simulation."""

    def __init__(
        self,
        strategies:   Dict[str, object],
        risk_manager: Optional[object] = None,
        coin:         str = "BTC",
        oi_schedule:  Optional[Callable[[int], float]] = None,
    ) -> None:
        self.strategies   = strategies
        self.risk_manager = risk_manager
        self.coin         = coin
        self.oi_schedule  = oi_schedule or (lambda _: 1_000_000.0)
        self._fill_sim    = PassiveFillSimulator()
        self._pending:    Dict[str, List[ExecutionOrder]] = defaultdict(list)
        self._metrics:    Dict[str, StrategyMetrics] = {n: StrategyMetrics() for n in strategies}
        self._inventory:  Dict[str, InventoryState]  = {n: InventoryState()  for n in strategies}
        self._fills:      List[SimulatedFill]         = []
        self._equity:     List[Tuple[float, float]]   = []
        self._cum_pnl:    float = 0.0

    def run(self, ticks: List[HistoricalTick]) -> BacktestReport:
        t0 = time.perf_counter()
        for i, tick in enumerate(ticks):
            self._step(i, tick)
            if i % 500 == 0 and i:
                logger.info("[%d/%d] pnl=%.4f fills=%d", i, len(ticks), self._cum_pnl, len(self._fills))
        dur = time.perf_counter() - t0
        report = BacktestReport(self._metrics, self._equity, self._fills, len(ticks), dur, self.coin)
        logger.info("Done %.2fs PnL=%.4f DD=%.4f Sharpe=%.3f", dur, report.total_pnl, report.max_drawdown, report.sharpe)
        return report

    def _step(self, idx: int, tick: HistoricalTick) -> None:
        book, trades = tick.book, tick.trades
        oi = self.oi_schedule(idx)

        for name, orders in list(self._pending.items()):
            filled, remaining = [], []
            for o in orders:
                f = self._fill_sim.simulate(o, trades, book)
                (filled if f else remaining).append(f or o)
            self._pending[name] = [o for o in remaining if isinstance(o, ExecutionOrder)]
            self._commit_fills(name, [(o, f) for o, f in zip(orders, filled)
                                      if isinstance(f, SimulatedFill)], book)

        for name, strat in self.strategies.items():
            try:
                result = _call_strategy(strat, book, trades, oi)
            except Exception as exc:
                logger.error("Strategy %s tick %d: %s", name, idx, exc)
                continue
            orders = self._check(name, _extract_orders(result), book)
            self._metrics[name].order_count += len(orders)
            self._pending[name].extend(orders)

        self._equity.append((tick.timestamp, self._cum_pnl))

    def _check(self, name, orders, book):
        if self.risk_manager is None:
            return orders
        from init import MarketRegime
        self.risk_manager.update_market_state(book.mid)
        return self.risk_manager.pre_flight_check(
            name, orders, self._inventory[name], 0.001, MarketRegime.TRENDING
        )

    def _commit_fills(self, name, filled, book):
        for _, fill in filled:
            inv = self._inventory[name]
            m   = self._metrics[name]
            inv.update_on_fill(fill.side, fill.fill_size, fill.fill_price, book.mid, True)
            m.spread_pnl            += abs(fill.fill_price - fill.mid_at_fill) * fill.fill_size
            m.realized_pnl           = inv.realized_pnl
            m.adverse_selection_cost = inv.adverse_selection_cost
            m.fill_count            += 1
            self._fills.append(fill)
        self._cum_pnl = sum(i.realized_pnl for i in self._inventory.values())

class ProBacktestEngine:
    """FIFO queue position + probabilistic cancels + configurable feed/order latency."""

    def __init__(
        self,
        strategies:     Dict[str, object],
        risk_manager:   Optional[object] = None,
        coin:           str = "BTC",
        oi_schedule:    Optional[Callable[[int], float]] = None,
        latency_config: Optional[LatencyConfig] = None,
        cancel_model:   Optional[object] = None,
        seed:           int = 0,
    ) -> None:
        self.strategies   = strategies
        self.risk_manager = risk_manager
        self.coin         = coin
        self.oi_schedule  = oi_schedule or (lambda _: 1_000_000.0)

        self._rng     = np.random.default_rng(seed)
        self._lat_cfg = latency_config or LatencyConfig()

        self._queue_sim  = FIFOQueueSimulator(cancel_model or ReduceRatioCancelModel())
        self._lat_sim    = LatencySimulator(self._lat_cfg, self._rng)
        self._metrics:   Dict[str, StrategyMetrics] = {n: StrategyMetrics() for n in strategies}
        self._inventory: Dict[str, InventoryState]  = {n: InventoryState()  for n in strategies}
        self._fills:     List[SimulatedFill]         = []
        self._equity:    List[Tuple[float, float]]   = []
        self._cum_pnl:   float = 0.0

        # Map order_id → strategy name for fill attribution
        self._order_owner: Dict[str, str] = {}

    def run(self, ticks: List[HistoricalTick]) -> BacktestReport:
        t0 = time.perf_counter()
        logger.info("ProBacktest starting: %d ticks lat_feed=%.0fμs lat_order=%.0fμs",
                    len(ticks), self._lat_cfg.feed_latency_us, self._lat_cfg.order_latency_us)

        for idx, tick in enumerate(ticks):
            self._step(idx, tick)
            if idx % 500 == 0 and idx:
                logger.info("[%d/%d] pnl=%.4f fills=%d in_flight=%d queue_active=%d",
                            idx, len(ticks), self._cum_pnl, len(self._fills),
                            self._lat_sim.in_flight_count(), self._queue_sim.active_count())

        dur = time.perf_counter() - t0
        report = BacktestReport(
            self._metrics, self._equity, self._fills,
            len(ticks), dur, self.coin, mode="pro",
        )
        logger.info("ProBacktest done %.2fs PnL=%.4f DD=%.4f Sharpe=%.3f",
                    dur, report.total_pnl, report.max_drawdown, report.sharpe)
        return report

    def _step(self, idx: int, tick: HistoricalTick) -> None:
        now    = tick.timestamp
        book   = tick.book
        trades = tick.trades
        oi     = self.oi_schedule(idx)

        self._lat_sim.record_book(now, book)

        for pending in self._lat_sim.poll_arrived(now):
            self._register_arrived_order(pending, book, now)

        fifo_fills = self._queue_sim.process_tick(book, trades)
        for state, fill_size in fifo_fills:
            self._commit_fifo_fill(state, fill_size, book, now)

        stale = self._lat_sim.stale_book(now) or book
        for name, strat in self.strategies.items():
            try:
                result = _call_strategy(strat, stale, trades, oi)
            except Exception as exc:
                logger.error("Strategy %s tick %d: %s", name, idx, exc)
                continue

            new_orders = _extract_orders(result)
            if self.risk_manager is not None:
                self.risk_manager.update_market_state(stale.mid)
                from init import MarketRegime
                new_orders = self.risk_manager.pre_flight_check(
                    name, new_orders, self._inventory[name], 0.001, MarketRegime.TRENDING
                )

            for order in new_orders:
                arrival_ts = self._lat_sim.schedule(order, name, now, stale)
                self._order_owner[order.client_id] = name
                latency_us = (arrival_ts - now) * 1e6
                self._metrics[name].order_count += 1
                self._metrics[name].record_latency(latency_us)

        self._equity.append((now, self._cum_pnl))

    def _register_arrived_order(
        self,
        pending: _PendingOrder,
        book:    OrderBook,
        now:     float,
    ) -> None:
        order    = pending.order
        strategy = pending.strategy
        levels   = book.bids if order.side == Side.BUY else book.asks
        depth    = next(
            (lvl.size for lvl in levels if abs(lvl.price - order.price) < 0.01),
            0.0
        )
        state = FIFOQueueState(
            order_id=order.client_id,
            strategy=strategy,
            side=order.side,
            price=order.price,
            remaining=order.size,
            qty_in_front=depth,
            submitted_ts=pending.decision_book.timestamp,
            arrival_ts=now,
        )
        self._queue_sim.register(state)

        # Log queue displacement from latency
        decision_depth = next(
            (lvl.size for lvl in (
                pending.decision_book.bids if order.side == Side.BUY
                else pending.decision_book.asks
            ) if abs(lvl.price - order.price) < 0.01),
            0.0,
        )
        if depth > decision_depth + 1e-6:
            logger.debug(
                "LATENCY DISPLACEMENT [%s] %.1f: decision_depth=%.2f arrival_depth=%.2f (+%.2f)",
                strategy, order.price, decision_depth, depth, depth - decision_depth,
            )

    def _commit_fifo_fill(
        self,
        state:     FIFOQueueState,
        fill_size: float,
        book:      OrderBook,
        now:       float,
    ) -> None:
        name = state.strategy
        inv  = self._inventory[name]
        m    = self._metrics[name]

        inv.update_on_fill(state.side, fill_size, state.price, book.mid, is_maker=True)

        fill = SimulatedFill(
            strategy=name,
            order_id=state.order_id,
            side=state.side,
            fill_price=state.price,
            fill_size=fill_size,
            mid_at_fill=book.mid,
            timestamp=now,
            is_maker=True,
            queue_model="fifo",
            qty_in_front_at_fill=state.qty_in_front,
        )
        self._fills.append(fill)

        m.spread_pnl            += abs(state.price - book.mid) * fill_size
        m.realized_pnl           = inv.realized_pnl
        m.adverse_selection_cost = inv.adverse_selection_cost
        m.fill_count            += 1
        self._cum_pnl = sum(i.realized_pnl for i in self._inventory.values())

        # Track cumulative queue advancement for diagnostics
        m.total_queue_advancement += max(0.0, state.qty_in_front)

def run_synthetic_backtest(
    n_ticks: int   = 1_000,
    mid:     float = 50_000.0,
    seed:    int   = 42,
    pro:     bool  = False,
    latency_config: Optional[LatencyConfig] = None,
    cancel_model:   Optional[object] = None,
) -> BacktestReport:
    from predatory_liquidity import PredatoryLiquidityStrategy
    from adaptive_guerrilla import AdaptiveGuerrillaStrategy
    from liquidation_frontrun import LiquidationFrontrunStrategy
    from central_risk_manager import CentralRiskManager, RiskConfig

    strategies = {
        "predatory_liq": PredatoryLiquidityStrategy(),
        "adaptive_gmm":  AdaptiveGuerrillaStrategy(),
        "liq_frontrun":  LiquidationFrontrunStrategy(),
    }
    rm = CentralRiskManager(RiskConfig(
        max_net_position=5.0, max_drawdown_limit=2000.0, daily_loss_limit=5000.0,
    ))
    ticks = TickLoader.synthetic(n_ticks=n_ticks, mid0=mid, seed=seed)

    if pro:
        engine = ProBacktestEngine(
            strategies=strategies, risk_manager=rm,
            latency_config=latency_config or LatencyConfig(),
            cancel_model=cancel_model or ReduceRatioCancelModel(),
            seed=seed,
        )
    else:
        engine = BacktestEngine(strategies=strategies, risk_manager=rm)

    return engine.run(ticks)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    print("── Simple mode ──")
    run_synthetic_backtest(n_ticks=500, pro=False).print_summary()

    print("── Pro mode (FIFO + latency) ──")
    run_synthetic_backtest(
        n_ticks=500, pro=True,
        latency_config=LatencyConfig(feed_latency_us=150, order_latency_us=600),
        cancel_model=ProbQueueCancelModel(),
    ).print_summary()
