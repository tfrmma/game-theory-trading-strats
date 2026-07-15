"""
Central runner simulation and live Hyperliquid modes.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict
from typing import List
import time

import numpy as np

from engine.init import (
    simulate_order_book, simulate_trade_tape, Side, OrderBook, Trade,
    MarketRegime, compute_realized_volatility,
)
from strategies.spoofing_counter import SpoofingDetector, SpoofCounterStrategy
from strategies.predatory_liquidity import PredatoryLiquidityStrategy
from strategies.info_asymmetry import FlowToxicityClassifier
from strategies.queue_warfare import QueueWarfareStrategy
from strategies.funding_arbitrage import FundingArbitrageStrategy, FundingSnapshot
from strategies.liquidation_frontrun import LiquidationFrontrunStrategy
from strategies.adaptive_guerrilla import AdaptiveGuerrillaStrategy
from engine.hyperliquid_feed import HyperliquidFeed
from engine.central_risk_manager import CentralRiskManager, RiskConfig

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

        # RiskConfig values below are the example from README.md's "Risk manager"
        # section, not a calibrated production config. max_net_position/drawdown/
        # daily_loss_limit/size_decimals must be reviewed per venue before this
        # gates real orders. This is a global circuit breaker + pre-trade gate
        # that was previously built (central_risk_manager.py) but never wired
        # into the runner signals bypassed it entirely.
        self.risk_manager = CentralRiskManager(RiskConfig(
            max_net_position    = 5.0,
            max_drawdown_limit  = 500.0,
            daily_loss_limit    = 1000.0,
            size_decimals       = 4,
            toxicity_cooldown_s = 60.0,
        ))

        self.results       = defaultdict(list)
        self.signals_log   = []
        self.strategy_pnl  = {name: 0.0 for name in self.strategies}

        self.mid           = 50_000.0
        self.oi            = 1_000_000.0
        self.current_book: OrderBook | None = None
        self.feed          = None

        self._mid_history: List[float] = []
        self._last_toxicity_state: dict = {}

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

            book = self.current_book
            self._mid_history.append(book.mid)
            self._last_toxicity_state = self.toxicity.update(book, trades)
            self.risk_manager.update_market_state(book.mid)

            realized_pnl   = sum(self.strategy_pnl.values())
            unrealized_pnl = sum(
                s.inventory.unrealized_pnl for s in self.strategies.values() if hasattr(s, "inventory")
            )
            alive = self.risk_manager.update_global_pnl(realized_pnl, unrealized_pnl)
            if not alive:
                logger.critical("RISK HALTED (live): %s", self.risk_manager.halt_reason)
                return

            for name, strat in self.strategies.items():
                if name == "spoof_counter":
                    for ev in self.detector.update(book):
                        signal = strat.process_spoof_event(ev)
                        if signal:
                            self.signals_log.append((name, signal))

                elif name == "predatory_liquidity":
                    signal = strat.update(book, trades)
                    if signal:
                        self.signals_log.append((name, signal))

                elif name == "queue_warfare":
                    strat.update(book, trades)

                elif name == "liq_frontrun":
                    actions = strat.update(book, trades, self.oi)
                    if actions.get("taker_entry"):
                        self.signals_log.append((name, actions["taker_entry"]))
                    self.strategy_pnl[name] = strat.total_pnl

                elif name == "adaptive_guerrilla":
                    orders, cancels = strat.update(book, trades)
                    if orders:
                        regime = self._infer_regime()
                        vol    = self._current_volatility()
                        approved = self.risk_manager.pre_flight_check(
                            strategy_name=name,
                            proposed_orders=orders,
                            current_inventory=strat.inventory,
                            current_volatility=vol,
                            market_regime=regime,
                        )
                        if len(approved) < len(orders):
                            logger.info(
                                "RISK (live): %d/%d guerrilla orders passed pre-flight",
                                len(approved), len(orders),
                            )
                        self._reconcile_rejected_quotes(strat, orders, approved)
                    if hasattr(strat, "inventory"):
                        self.strategy_pnl[name] = strat.inventory.realized_pnl

        async def funding_poller() -> None:
            # Hyperliquid funding settles hourly (docs) polling once a minute
            # is far more than enough resolution and stays well inside rate limits.
            strat = self.strategies["funding_arb"]
            while True:
                ctx = self.feed.get_funding_ctx()
                if ctx is not None:
                    now = time.time()
                    next_hour_ts = (now // 3600 + 1) * 3600  # top of next UTC hour
                    snapshot = FundingSnapshot(
                        rate=ctx["rate"],
                        next_funding_ts=next_hour_ts,
                        open_interest=ctx["open_interest"],
                        mark_price=ctx["mark_price"],
                        index_price=ctx["oracle_price"],
                        timestamp=now,
                    )
                    strat.update_funding(snapshot, [])
                    entry_signal = strat.evaluate_entry(snapshot)
                    if entry_signal:
                        self.signals_log.append(("funding_arb", entry_signal))
                    exit_reason = strat.evaluate_exit(snapshot)
                    if exit_reason:
                        pnl_summary = strat.close_position(snapshot, exit_reason)
                        self.strategy_pnl["funding_arb"] += pnl_summary.get("total_pnl", 0.0)
                await asyncio.sleep(60.0)

        await self.feed.start(book_handler, trade_handler)
        poller_task = asyncio.create_task(funding_poller())

        try:
            tick = 0
            while True:
                await asyncio.sleep(0.2)
                if tick % 50 == 0:
                    self._print_progress(tick)
                tick += 1
        except asyncio.CancelledError:
            poller_task.cancel()
            await self.feed.stop()

    @staticmethod
    def _reconcile_rejected_quotes(strat, proposed, approved) -> None:
        """When pre_flight_check shaves/rejects a guerrilla order, retract the
        matching quote from the strategy's own _active_quotes/_cancel_log so its
        internal fill simulation doesn't act on size the risk manager didn't
        approve. ExecutionOrder and GuerrillaQuote don't share an id (they're
        built side by side in _generate_quotes with independent ids), so this
        matches on (side, price) the only fields the two objects share.
        Uses the strategy's own existing cancellation pattern (_cancel_log +
        del _active_quotes[qid]), the same one it already uses for TOXIC_FLOW
        cancels not a new mechanism."""
        if len(approved) >= len(proposed):
            return
        approved_keys = {(o.side, round(o.price, 1)) for o in approved}
        rejected = [o for o in proposed if (o.side, round(o.price, 1)) not in approved_keys]
        for order in rejected:
            for qid, quote in list(strat._active_quotes.items()):
                if quote.side == order.side and round(quote.price, 1) == round(order.price, 1):
                    strat._cancel_log.append({"id": qid, "reason": "RISK_REJECTED"})
                    del strat._active_quotes[qid]
                    break

    def _process_tick(self, book: OrderBook, trades: List[Trade], tick: int) -> None:
        self._mid_history.append(book.mid)
        # Previously this return value was discarded VPIN/Kyle's lambda/regime
        # were computed every tick and thrown away. Now captured for _infer_regime().
        self._last_toxicity_state = self.toxicity.update(book, trades)

        # --- Central risk manager: global circuit breaker, evaluated every tick ---
        self.risk_manager.update_market_state(book.mid)
        realized_pnl   = sum(self.strategy_pnl.values())
        # InventoryState.unrealized_pnl now does real mark-to-market (init.py)
        # only adaptive_guerrilla carries one today. Reflects the mark price as
        # of that strategy's last fill, not a continuously live mark.
        unrealized_pnl = sum(
            s.inventory.unrealized_pnl for s in self.strategies.values() if hasattr(s, "inventory")
        )
        alive = self.risk_manager.update_global_pnl(realized_pnl, unrealized_pnl)
        if not alive:
            if tick % 50 == 0:
                print(f"[TICK {tick:4d}] \U0001F6D1 RISK HALTED: {self.risk_manager.halt_reason}")
            return  # no strategy runs while halted matches CentralRiskManager's own contract

        for name, strat in self.strategies.items():
            pnl_before = self.strategy_pnl[name]

            if name == "spoof_counter":
                for ev in self.detector.update(book):
                    signal = strat.process_spoof_event(ev)
                    if signal:
                        self.signals_log.append((name, signal))
                # No sizing/exit model exists for this strategy's Signal output yet,
                # so there's no honest way to derive a $ PnL from it. Reporting 0.0
                # instead of the previous np.random.normal(8, 4) noise.

            elif name == "predatory_liquidity":
                signal = strat.update(book, trades)
                if signal:
                    self.signals_log.append((name, signal))
                # Same limitation as spoof_counter Signal has entry/target/stop
                # but no size, order_type, or exit trigger. No PnL fabricated.

            elif name == "queue_warfare":
                cancel_ids = strat.update(book, trades)
                if cancel_ids:
                    self.signals_log.append((name, f"{len(cancel_ids)} cancel(s)"))
                # update() returns cancel-order ids, not a directional signal —
                # there is nothing to mark PnL against. No PnL fabricated.

            elif name == "funding_arb":
                # Rate magnitude matches funding_arbitrage.py's own
                # simulate_funding_arb() demo (base_rate=0.0008) rather than
                # the previous 0.0000125-0.000075, which sat an order of
                # magnitude below EXTREME_FUNDING_THRESHOLD (0.0005) and could
                # never actually trigger evaluate_entry().
                rate_decay = 1 - min(tick / self.n_ticks, 1.0) * 0.6
                snapshot = FundingSnapshot(
                    rate=0.0008 * rate_decay,
                    next_funding_ts=time.time() + 3600,
                    open_interest=self.oi,
                    mark_price=self.mid,
                    index_price=self.mid * 0.998,
                    timestamp=time.time(),
                )
                strat.update_funding(snapshot, trades)
                # evaluate_entry/evaluate_exit/close_position already existed and
                # already compute real funding+perp+spot PnL they were just never
                # called from the runner. Wiring them in the same order the
                # module's own simulate_funding_arb() demo uses.
                entry_signal = strat.evaluate_entry(snapshot)
                if entry_signal:
                    self.signals_log.append((name, entry_signal))
                exit_reason = strat.evaluate_exit(snapshot)
                if exit_reason:
                    pnl_summary = strat.close_position(snapshot, exit_reason)
                    self.strategy_pnl[name] += pnl_summary.get("total_pnl", 0.0)

            elif name == "liq_frontrun":
                actions = strat.update(book, trades, self.oi)
                if actions.get("taker_entry"):
                    self.signals_log.append((name, actions["taker_entry"]))
                # total_pnl was already tracked internally (self._trade_log in
                # liquidation_frontrun.py) and ignored by the runner in favor of
                # np.random.normal(18, 6). Reading the real property instead.
                self.strategy_pnl[name] = strat.total_pnl

            elif name == "adaptive_guerrilla":
                orders, cancels = strat.update(book, trades)
                if orders:
                    regime = self._infer_regime()
                    vol    = self._current_volatility()
                    approved = self.risk_manager.pre_flight_check(
                        strategy_name=name,
                        proposed_orders=orders,
                        current_inventory=strat.inventory,
                        current_volatility=vol,
                        market_regime=regime,
                    )
                    if len(approved) < len(orders):
                        logger.info(
                            "RISK: %d/%d guerrilla orders passed pre-flight "
                            "(regime=%s vol=%.6f)",
                            len(approved), len(orders), regime.name, vol,
                        )
                    # adaptive_guerrilla._generate_quotes() commits accepted
                    # quotes into its own internal _active_quotes before
                    # returning them (that's what its passive-fill simulation
                    # reads from). A pre_flight_check reject/shave now retracts
                    # the matching quote via the strategy's own existing
                    # cancellation pattern see _reconcile_rejected_quotes.
                    self._reconcile_rejected_quotes(strat, orders, approved)
                if hasattr(strat, "inventory"):
                    self.strategy_pnl[name] = strat.inventory.realized_pnl

            self.results[name].append(self.strategy_pnl[name] - pnl_before)

    def _current_volatility(self) -> float:
        """Realized vol from the runner's own mid-price history, via init.py's
        existing compute_realized_volatility() previously never called anywhere
        in the repo despite being written for exactly this purpose."""
        if len(self._mid_history) < 20:
            return 0.0
        window = np.array(self._mid_history[-100:])
        return compute_realized_volatility(window)

    def _infer_regime(self) -> MarketRegime:
        """Maps FlowToxicityClassifier's regime string (SAFE/CAUTION/TOXIC/
        EXTREME_TOXICITY) onto CentralRiskManager's MarketRegime enum.
        No classifier for TRENDING vs MEAN_REVERT vs ILLIQUID exists anywhere
        in this repo today tick_by_tick_backtester.py hardcodes
        MarketRegime.TRENDING wherever it calls pre_flight_check, and this
        does the same as the default. Only the TOXIC branch is grounded in
        real signal (VPINCalculator's existing, already-calibrated thresholds)."""
        regime_str = self._last_toxicity_state.get("regime", "SAFE")
        if regime_str in ("TOXIC", "EXTREME_TOXICITY"):
            return MarketRegime.TOXIC
        return MarketRegime.TRENDING

    def _print_progress(self, tick: int) -> None:
        print(f"[TICK {tick:4d}] MID={self.mid:,.0f} | Signals={len(self.signals_log)}")

    # Strategies whose PnL is derived from a real fill/execution model.
    # The rest generate Signal/action objects with no sizing or exit logic
    # attached (see comments in _process_tick) their 0.00 means "not modeled
    # yet", not "modeled and flat".
    _PNL_TRACKED = {"adaptive_guerrilla", "liq_frontrun", "funding_arb"}

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
            marker    = "" if name in self._PNL_TRACKED else " *"
            print(f"{name:<22} {total_pnl:12.2f}{marker:<2}{avg_pnl:10.3f} {drawdown:12.2f} {signals:8d}")

        untracked = sorted(n for n in self.strategies if n not in self._PNL_TRACKED)
        if untracked:
            print(f"\n* No fill/sizing model yet PnL not modeled (0.00 != flat): {', '.join(untracked)}")
        print(f"\n🛑 halted={self.risk_manager.is_halted}"
              f" | session_pnl={self.risk_manager.session_pnl:.2f}"
              f" | drawdown={self.risk_manager.drawdown:.2f}")
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
