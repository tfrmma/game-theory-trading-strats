"""
Tests for liquidation_frontrun.py. total_pnl was already tracked internally
(self._trade_log) but runner.py ignored it in favor of np.random.normal(18, 6)
before this session's fix these tests pin that property and the taker-exit
PnL math it's built from.
"""
from __future__ import annotations

import numpy as np
import pytest

from liquidation_frontrun import CascadeDetector, LiquidationFrontrunStrategy
from init import Trade, Side


def make_trade(price=100.0, size=1.0, side=Side.BUY) -> Trade:
    return Trade(price=price, size=size, side=side, timestamp=0.0, is_liquidation=False, trade_id="t")


class TestTotalPnl:
    def test_zero_with_no_trades(self):
        strat = LiquidationFrontrunStrategy()
        assert strat.total_pnl == 0.0

    def test_sums_trade_log_entries(self):
        strat = LiquidationFrontrunStrategy()
        strat._trade_log.append({"type": "taker", "pnl": 42.5, "reason": "test"})
        strat._trade_log.append({"type": "taker", "pnl": -12.0, "reason": "test"})
        assert strat.total_pnl == pytest.approx(30.5)

    def test_runner_read_pattern_reflects_new_trades_immediately(self):
        """runner.py does self.strategy_pnl[name] = strat.total_pnl (an
        absolute reassignment, not +=) every tick — confirms that pattern
        stays correct as more trades close."""
        strat = LiquidationFrontrunStrategy()
        strat._trade_log.append({"type": "taker", "pnl": 10.0, "reason": "a"})
        snapshot_1 = strat.total_pnl
        strat._trade_log.append({"type": "taker", "pnl": 5.0, "reason": "b"})
        snapshot_2 = strat.total_pnl
        assert snapshot_1 == pytest.approx(10.0)
        assert snapshot_2 == pytest.approx(15.0)


class TestTakerExitPnl:
    def test_long_taker_exit_pnl_sign(self):
        strat = LiquidationFrontrunStrategy()
        strat._taker_position = {
            "side": Side.BUY, "entry_price": 100.0, "size": 1.0,
            "target": 999.0, "stop": 0.0,  # far away — exit triggers via cascade_exhausting only
        }
        actions = {"maker_cancels": []}
        strat._manage_taker_exit(mid=110.0, cascade_exhausting=True, actions=actions)
        assert strat.total_pnl == pytest.approx(10.0)
        assert strat._taker_position is None

    def test_short_taker_exit_pnl_sign(self):
        strat = LiquidationFrontrunStrategy()
        strat._taker_position = {
            "side": Side.SELL, "entry_price": 100.0, "size": 2.0,
            "target": 0.0, "stop": 999.0,
        }
        actions = {"maker_cancels": []}
        strat._manage_taker_exit(mid=90.0, cascade_exhausting=True, actions=actions)
        assert strat.total_pnl == pytest.approx(20.0)  # short profits as price falls


class TestCascadeDetector:
    def test_insufficient_data_returns_false(self):
        det = CascadeDetector()
        starting, exhausting, desc = det.update(trades=[make_trade()], oi=1_000_000.0, mid=100.0)
        assert starting is False
        assert exhausting is False

    def test_steady_state_does_not_flag_cascade(self):
        det = CascadeDetector(volume_spike_multiple=3.0, oi_drop_threshold_pct=0.005)
        rng = np.random.default_rng(0)
        starting = False
        for _ in range(15):
            trades = [make_trade(size=abs(rng.normal(1.0, 0.1))) for _ in range(5)]
            starting, _, _ = det.update(trades=trades, oi=1_000_000.0, mid=100.0)
        assert starting is False

    def test_volume_spike_plus_oi_drop_flags_cascade_starting(self):
        det = CascadeDetector(volume_spike_multiple=3.0, oi_drop_threshold_pct=0.005)
        # Baseline: steady, small volume, flat OI
        for _ in range(12):
            det.update(trades=[make_trade(size=1.0)], oi=1_000_000.0, mid=100.0)
        # Spike: much larger volume + OI dropping sharply (liquidations unwinding)
        starting = False
        for i in range(3):
            trades = [make_trade(size=20.0) for _ in range(10)]
            starting, _, _ = det.update(trades=trades, oi=1_000_000.0 - (i + 1) * 20_000.0, mid=95.0)
        assert starting is True
