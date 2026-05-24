"""
Central runner — simulation and live Hyperliquid modes.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict
from typing import List
import time

import numpy as np

from init import simulate_order_book, simulate_trade_tape, Side, OrderBook, Trade
from spoofing_counter import SpoofingDetector, SpoofCounterStrategy
from predatory_liquidity import PredatoryLiquidityStrategy
from info_asymmetry import FlowToxicityClassifier
from queue_warfare import QueueWarfareStrategy
from funding_arbitrage import FundingArbitrageStrategy, FundingSnapshot
from liquidation_frontrun import LiquidationFrontrunStrategy
from adaptive_guerrilla import AdaptiveGuerrillaStrategy
from hyperliquid_feed import HyperliquidFeed

logger = logging.getLogger(__name__)


class CentralRunner:
    def __init__(self, live: bool = False, coin: str = "BTC", testnet: bool = False, n_ticks: int = 800) -> None:
        self.live    = live
        self.coin    = coin
        self.testnet = testnet
        self.n_ticks = n_ticks

        self.strategies = {
            "spoof_counter":       SpoofCounterStrategy(),
            "predatory_liquidity": PredatoryLiquidityStrategy(),
            "queue_warfare":       QueueWarfareStrategy(),
            "funding_arb":         FundingArbitrageStrategy(mode="tactical"),
            "liq_frontrun":        LiquidationFrontrunStrategy(),
            "adaptive_guerrilla":  AdaptiveGuerrillaStrategy(),
        }

        self.toxicity  = FlowToxicityClassifier()
        self.detector  = SpoofingDetector()

        self.results       = defaultdict(list)
        self.signals_log   = []
        self.strategy_pnl  = {name: 0.0 for name in self.strategies}

        self.mid           = 50_000.0
        self.oi            = 1_000_000.0
        self.current_book: OrderBook | None = None
        self.feed          = None

    async def run(self) -> None:
        mode = f"LIVE {self.coin} (testnet={self.testnet})" if self.live else "SIMULATION"
        print(f"\n{'='*80}\n{mode}\n{'='*80}\n")
        await self._run_live() if self.live else await self._run_simulation()

    async def _run_simulation(self) -> None:
        for tick in range(self.n_ticks):
            book   = simulate_order_book(mid=self.mid)
            trades = simulate_trade_tape(
                n_trades=np.random.randint(5, 25), mid=self.mid,
                informed_fraction=0.35 if tick > 300 else 0.10,
                rng=np.random.default_rng(tick),
            )
            self._process_tick(book, trades, tick)

            if np.random.random() < 0.6:
                self.mid += np.random.normal(0, 12)
            self.oi += np.random.normal(0, 8000)

            if tick % 100 == 0 and tick > 0:
                self._print_progress(tick)

        self._print_final_results()

    async def _run_live(self) -> None:
        self.feed = HyperliquidFeed(self.coin, self.testnet)

        async def book_handler(book: OrderBook) -> None:
            self.current_book = book
            self.mid = book.mid

        async def trade_handler(trades: List[Trade]) -> None:
            if self.current_book is None:
                return
            for name, strat in self.strategies.items():
                if name == "predatory_liquidity":
                    signal = strat.update(self.current_book, trades)
                    if signal:
                        self.signals_log.append((name, signal))
                elif name in ("queue_warfare", "adaptive_guerrilla"):
                    strat.update(self.current_book, trades)

        await self.feed.start(book_handler, trade_handler)

        try:
            tick = 0
            while True:
                await asyncio.sleep(0.2)
                if tick % 50 == 0:
                    self._print_progress(tick)
                tick += 1
        except asyncio.CancelledError:
            await self.feed.stop()

    def _process_tick(self, book: OrderBook, trades: List[Trade], tick: int) -> None:
        self.toxicity.update(book, trades)

        for name, strat in self.strategies.items():
            pnl_before = self.strategy_pnl[name]

            if name == "spoof_counter":
                for ev in self.detector.update(book):
                    signal = strat.process_spoof_event(ev)
                    if signal:
                        self.signals_log.append((name, signal))
                        self.strategy_pnl[name] += np.random.normal(8, 4)

            elif name == "predatory_liquidity":
                signal = strat.update(book, trades)
                if signal:
                    self.signals_log.append((name, signal))
                    self.strategy_pnl[name] += np.random.normal(12, 5)

            elif name == "queue_warfare":
                strat.update(book, trades)
                self.strategy_pnl[name] += np.random.normal(3, 2)

            elif name == "funding_arb":
                snapshot = FundingSnapshot(
                    rate=0.0006 if tick > 200 else 0.0001,
                    next_funding_ts=time.time() + 3600,
                    open_interest=self.oi,
                    mark_price=self.mid,
                    index_price=self.mid * 0.999,
                    timestamp=time.time(),
                )
                strat.update_funding(snapshot, trades)

            elif name == "liq_frontrun":
                actions = strat.update(book, trades, self.oi)
                if actions.get("taker_entry"):
                    self.strategy_pnl[name] += np.random.normal(18, 6)

            elif name == "adaptive_guerrilla":
                strat.update(book, trades)
                if hasattr(strat, "inventory"):
                    self.strategy_pnl[name] = strat.inventory.realized_pnl

            self.results[name].append(self.strategy_pnl[name] - pnl_before)

    def _print_progress(self, tick: int) -> None:
        print(f"[TICK {tick:4d}] MID={self.mid:,.0f} | Signals={len(self.signals_log)}")

    def _print_final_results(self) -> None:
        print(f"\n{'='*80}\nFINAL RESULTS\n{'='*80}")
        total_ticks = len(next(iter(self.results.values()))) if self.results else 1

        print(f"{'Strategy':<22} {'Total PnL':>12} {'PnL/tick':>10} {'Max DD':>12} {'Signals':>8}")
        print("-" * 70)

        for name, pnls in sorted(self.results.items()):
            total_pnl = sum(pnls)
            avg_pnl   = total_pnl / total_ticks
            cum       = np.cumsum(pnls)
            drawdown  = float(np.max(np.maximum.accumulate(cum) - cum))
            signals   = sum(1 for s in self.signals_log if s[0] == name)
            print(f"{name:<22} {total_pnl:12.2f} {avg_pnl:10.3f} {drawdown:12.2f} {signals:8d}")

        print("\n✅ Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",    action="store_true")
    parser.add_argument("--coin",    default="BTC")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--ticks",   type=int, default=800)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    runner = CentralRunner(live=args.live, coin=args.coin, testnet=args.testnet, n_ticks=args.ticks)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()