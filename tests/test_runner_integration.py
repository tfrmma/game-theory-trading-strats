"""
Integration tests for runner.py the piece that wires every strategy
together. Also covers two regressions fixed this session: RL_tuner.py's
broken import, and the deque-slicing crash in CentralRiskManager.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

import runner as runner_mod
from init import ExecutionOrder, OrderType, Side, simulate_order_book, simulate_trade_tape


class TestRegressions:
    def test_rl_tuner_imports_cleanly(self):
        """Was: ModuleNotFoundError: No module named 'backtester' RL_tuner.py
        imported from a module that had been renamed to
        tick_by_tick_backtester.py, breaking every entry point in the file."""
        import RL_tuner
        assert hasattr(RL_tuner, "train_agent")
        assert hasattr(RL_tuner, "TrainConfig")

    def test_readme_import_examples_actually_work(self):
        from tick_by_tick_backtester import ProBacktestEngine, TickLoader, LatencyConfig, ProbQueueCancelModel
        from RL_tuner import train_agent, compare_baseline, TrainConfig  # noqa: F401


class TestSimulationSmokeTest:
    def test_runs_many_ticks_without_crashing(self):
        r = runner_mod.CentralRunner(live=False, n_ticks=300)
        asyncio.run(r.run())
        # every strategy produced a results series
        assert set(r.results.keys()) == set(r.strategies.keys())
        for pnls in r.results.values():
            assert len(pnls) == 300

    def test_untracked_strategies_report_zero_not_random_noise(self):
        """Was: predatory_liquidity/queue_warfare/spoof_counter accumulated
        np.random.normal(...) every time they fired fabricated PnL with no
        underlying sizing/exit model. Two independent runs with different
        seeds used to diverge; now they're deterministically 0.0 since no
        fill model exists for these three yet."""
        results = []
        for seed in (1, 2):
            np.random.seed(seed)
            r = runner_mod.CentralRunner(live=False, n_ticks=150)
            asyncio.run(r.run())
            results.append(sum(r.results["predatory_liquidity"]))
        assert results[0] == 0.0
        assert results[1] == 0.0


class TestRiskManagerHaltsSimulation:
    def test_forced_loss_halts_all_strategies_immediately(self):
        r = runner_mod.CentralRunner(live=False, n_ticks=10)
        r.strategy_pnl["adaptive_guerrilla"] = -10_000.0  # far past any sane limit

        book = simulate_order_book(mid=r.mid)
        trades = simulate_trade_tape(10, r.mid, 0.1, rng=np.random.default_rng(1))
        signals_before = len(r.signals_log)

        r._process_tick(book, trades, tick=0)

        assert r.risk_manager.is_halted is True
        assert len(r.signals_log) == signals_before  # nothing ran this tick


class TestAdaptiveGuerrillaReconciliation:
    """Regression/behavior test for the fill-desync fix: adaptive_guerrilla
    commits quotes into its own _active_quotes before returning them, so a
    pre_flight_check rejection has to be reconciled back into that internal
    state or the strategy's fill simulation would act on size the risk
    manager never approved."""

    def test_rejected_quote_is_removed_from_active_quotes(self):
        r = runner_mod.CentralRunner(live=False, n_ticks=10)
        strat = r.strategies["adaptive_guerrilla"]

        kept_order     = ExecutionOrder(side=Side.BUY, price=100.0, size=0.1,
                                         order_type=OrderType.POST_ONLY, client_id="a")
        rejected_order = ExecutionOrder(side=Side.SELL, price=200.0, size=0.1,
                                         order_type=OrderType.POST_ONLY, client_id="b")

        from adaptive_guerrilla import GuerrillaQuote
        kept_quote     = GuerrillaQuote(side=Side.BUY, price=100.0, size=0.1, reservation_price=100.0)
        rejected_quote = GuerrillaQuote(side=Side.SELL, price=200.0, size=0.1, reservation_price=200.0)
        strat._active_quotes = {kept_quote.order_id: kept_quote, rejected_quote.order_id: rejected_quote}

        proposed = [kept_order, rejected_order]
        approved = [kept_order]  # simulate risk manager rejecting the second

        r._reconcile_rejected_quotes(strat, proposed, approved)

        remaining_prices = {q.price for q in strat._active_quotes.values()}
        assert 100.0 in remaining_prices
        assert 200.0 not in remaining_prices
        assert any(c["reason"] == "RISK_REJECTED" for c in strat._cancel_log)

    def test_nothing_removed_when_everything_approved(self):
        r = runner_mod.CentralRunner(live=False, n_ticks=10)
        strat = r.strategies["adaptive_guerrilla"]
        from adaptive_guerrilla import GuerrillaQuote
        quote = GuerrillaQuote(side=Side.BUY, price=100.0, size=0.1, reservation_price=100.0)
        strat._active_quotes = {quote.order_id: quote}

        order = ExecutionOrder(side=Side.BUY, price=100.0, size=0.1,
                                order_type=OrderType.POST_ONLY, client_id="a")
        r._reconcile_rejected_quotes(strat, [order], [order])

        assert len(strat._active_quotes) == 1
