"""
Tests for funding_arbitrage.py. FundingArbitrageStrategy.evaluate_entry/
evaluate_exit/close_position already existed and already computed real
funding+perp+spot PnL, but were never called anywhere in runner.py before
this session the runner only ever called update_funding(). These tests
pin the PnL math and the entry/exit gating that's now actually wired in.
"""
from __future__ import annotations

import time

import pytest

from strategies.funding_arbitrage import (
    EXTREME_FUNDING_THRESHOLD, FundingArbitragePosition, FundingArbitrageStrategy,
    FundingMonitor, FundingSnapshot,
)
from engine.init import Side


def make_snapshot(rate=0.0001, mark=50_000.0, index=50_000.0, next_funding_in=1800.0, oi=1_000_000.0):
    return FundingSnapshot(
        rate=rate, next_funding_ts=time.time() + next_funding_in,
        open_interest=oi, mark_price=mark, index_price=index, timestamp=time.time(),
    )


class TestFundingMonitorExtremeDetection:
    def test_not_extreme_below_threshold(self):
        mon = FundingMonitor()
        for _ in range(25):
            mon.update(make_snapshot(rate=0.00001))
        assert bool(mon.is_extreme_positive()) is False

    def test_extreme_positive_requires_both_rate_and_zscore(self):
        mon = FundingMonitor(z_score_extreme=2.0)
        # Feed a stable low baseline, then a spike far above EXTREME_FUNDING_THRESHOLD
        for _ in range(20):
            mon.update(make_snapshot(rate=0.00001))
        mon.update(make_snapshot(rate=EXTREME_FUNDING_THRESHOLD * 3))
        assert mon.current_rate == pytest.approx(EXTREME_FUNDING_THRESHOLD * 3)
        assert bool(mon.is_extreme_positive()) is True

    def test_extreme_negative_mirrors_positive(self):
        mon = FundingMonitor(z_score_extreme=2.0)
        for _ in range(20):
            mon.update(make_snapshot(rate=-0.00001))
        mon.update(make_snapshot(rate=-EXTREME_FUNDING_THRESHOLD * 3))
        assert bool(mon.is_extreme_negative()) is True
        assert bool(mon.is_extreme_positive()) is False


class TestClosePositionPnl:
    """Isolates the exact runner.py code path: evaluate_exit -> close_position
    -> accumulate total_pnl, given an already-open position."""

    def test_short_perp_long_spot_basis_capture(self):
        strat = FundingArbitrageStrategy(mode="tactical")
        strat._active_position = FundingArbitragePosition(
            perp_side=Side.SELL, perp_size=0.1, perp_entry=50_000.0,
            spot_size=0.1, spot_entry=49_900.0, entry_ts=time.time() - 700,
            target_exit_ts=time.time() - 1,
        )
        snap = make_snapshot(mark=50_100.0, index=49_950.0)
        exit_reason = strat.evaluate_exit(snap)
        assert exit_reason is not None

        pnl = strat.close_position(snap, exit_reason)
        # perp: short 0.1 @ 50000 -> 50100 = -10.0 ; spot: long 0.1 @ 49900 -> 49950 = +5.0
        assert pnl["perp_pnl"] == pytest.approx(-10.0)
        assert pnl["spot_pnl"] == pytest.approx(5.0)
        assert pnl["total_pnl"] == pytest.approx(pnl["perp_pnl"] + pnl["spot_pnl"] + pnl["funding_pnl"])

    def test_close_position_clears_active_position(self):
        strat = FundingArbitrageStrategy(mode="tactical")
        strat._active_position = FundingArbitragePosition(
            perp_side=Side.BUY, perp_size=0.05, perp_entry=100.0,
            spot_size=0.05, spot_entry=101.0, entry_ts=time.time() - 100,
            target_exit_ts=time.time() - 1,
        )
        snap = make_snapshot(mark=105.0, index=104.0)
        reason = strat.evaluate_exit(snap)
        strat.close_position(snap, reason)
        assert strat._active_position is None

    def test_runner_accumulation_pattern_sums_correctly_across_two_trades(self):
        """Mirrors exactly what runner.py does: strategy_pnl[name] +=
        pnl_summary['total_pnl'] on every close_position call."""
        strat = FundingArbitrageStrategy(mode="tactical")
        strategy_pnl = 0.0

        strat._active_position = FundingArbitragePosition(
            perp_side=Side.SELL, perp_size=0.1, perp_entry=100.0,
            spot_size=0.1, spot_entry=99.0, entry_ts=time.time() - 700,
            target_exit_ts=time.time() - 1,
        )
        snap1 = make_snapshot(mark=98.0, index=97.5)
        reason1 = strat.evaluate_exit(snap1)
        strategy_pnl += strat.close_position(snap1, reason1)["total_pnl"]

        strat._active_position = FundingArbitragePosition(
            perp_side=Side.BUY, perp_size=0.2, perp_entry=50.0,
            spot_size=0.2, spot_entry=50.5, entry_ts=time.time() - 700,
            target_exit_ts=time.time() - 1,
        )
        snap2 = make_snapshot(mark=52.0, index=51.0)
        reason2 = strat.evaluate_exit(snap2)
        strategy_pnl += strat.close_position(snap2, reason2)["total_pnl"]

        # perp1: short 0.1 100->98 = +0.2  ; spot1 (formula has no side flip): (97.5-99)*0.1 = -0.15
        # perp2: long  0.2 50->52  = +0.4  ; spot2: (51-50.5)*0.2 = +0.1
        expected = (0.2 - 0.15) + (0.4 + 0.1)
        assert strategy_pnl == pytest.approx(expected, abs=1e-6)


class TestEvaluateEntryRequiresPrintWindow:
    def test_no_entry_outside_pre_print_window(self):
        strat = FundingArbitrageStrategy(mode="tactical")
        for _ in range(21):
            strat.update_funding(make_snapshot(rate=EXTREME_FUNDING_THRESHOLD * 3, next_funding_in=99999), [])
        # next_funding_ts is far in the future -> outside PRE_PRINT_ENTRY_WINDOW_S
        snap = make_snapshot(rate=EXTREME_FUNDING_THRESHOLD * 3, next_funding_in=99999)
        assert strat.evaluate_entry(snap) is None

    def test_no_entry_when_already_in_a_position(self):
        strat = FundingArbitrageStrategy(mode="tactical")
        strat._active_position = FundingArbitragePosition(
            perp_side=Side.BUY, perp_size=0.1, perp_entry=100.0,
            spot_size=0.1, spot_entry=100.0, entry_ts=time.time(), target_exit_ts=time.time() + 700,
        )
        snap = make_snapshot(rate=EXTREME_FUNDING_THRESHOLD * 3, next_funding_in=600)
        assert strat.evaluate_entry(snap) is None
