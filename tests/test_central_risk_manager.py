"""
Tests for central_risk_manager.py the global circuit breaker + pre-trade
gate that runner.py now actually calls every tick.
"""
from __future__ import annotations

import time

import pytest

from engine.central_risk_manager import CentralRiskManager, RiskConfig
from engine.init import ExecutionOrder, InventoryState, MarketRegime, OrderType, Side


def make_config(**overrides) -> RiskConfig:
    defaults = dict(
        max_net_position=5.0,
        max_drawdown_limit=500.0,
        daily_loss_limit=1000.0,
        size_decimals=4,
        toxicity_cooldown_s=60.0,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def make_order(side=Side.BUY, price=50_000.0, size=1.0) -> ExecutionOrder:
    return ExecutionOrder(
        side=side, price=price, size=size, order_type=OrderType.POST_ONLY,
        client_id="test_order",
    )


class TestGlobalPnlCircuitBreaker:
    def test_alive_when_pnl_within_limits(self):
        rm = CentralRiskManager(make_config())
        assert rm.update_global_pnl(realized_pnl=10.0, unrealized_pnl=-5.0) is True
        assert rm.is_halted is False

    def test_halts_on_daily_loss_limit_breach(self):
        rm = CentralRiskManager(make_config(daily_loss_limit=100.0, max_drawdown_limit=10_000.0))
        alive = rm.update_global_pnl(realized_pnl=-150.0, unrealized_pnl=0.0)
        assert alive is False
        assert rm.is_halted is True
        assert "Daily loss limit" in rm.halt_reason

    def test_halts_on_drawdown_breach_even_if_still_net_positive(self):
        rm = CentralRiskManager(make_config(max_drawdown_limit=50.0, daily_loss_limit=10_000.0))
        rm.update_global_pnl(realized_pnl=200.0, unrealized_pnl=0.0)   # hwm now 200
        alive = rm.update_global_pnl(realized_pnl=140.0, unrealized_pnl=0.0)  # drawdown = 60 > 50
        assert alive is False
        assert "drawdown" in rm.halt_reason.lower()

    def test_once_halted_stays_halted_and_stops_updating_pnl(self):
        rm = CentralRiskManager(make_config(daily_loss_limit=10.0, max_drawdown_limit=10_000.0))
        rm.update_global_pnl(realized_pnl=-20.0, unrealized_pnl=0.0)
        assert rm.is_halted is True
        pnl_at_halt = rm.session_pnl
        # further calls should be no-ops (return False, don't update session_pnl)
        assert rm.update_global_pnl(realized_pnl=1_000_000.0, unrealized_pnl=0.0) is False
        assert rm.session_pnl == pnl_at_halt

    def test_unhalt_clears_state_and_allows_updates_again(self):
        rm = CentralRiskManager(make_config(daily_loss_limit=10.0, max_drawdown_limit=10_000.0))
        rm.update_global_pnl(realized_pnl=-20.0, unrealized_pnl=0.0)
        assert rm.is_halted is True
        rm.unhalt("manual review complete")
        assert rm.is_halted is False
        assert rm.halt_reason is None


class TestPreFlightCheckWhenHalted:
    def test_returns_empty_list_when_halted(self):
        rm = CentralRiskManager(make_config(daily_loss_limit=10.0, max_drawdown_limit=10_000.0))
        rm.update_global_pnl(realized_pnl=-20.0, unrealized_pnl=0.0)
        approved = rm.pre_flight_check(
            strategy_name="test", proposed_orders=[make_order()],
            current_inventory=InventoryState(), current_volatility=0.001,
            market_regime=MarketRegime.TRENDING,
        )
        assert approved == []


class TestFatFingerGuard:
    def test_rejects_order_exceeding_max_single_order_fraction(self):
        # max_single = max_net_position(5.0) * max_single_order_fraction(0.25) = 1.25
        rm = CentralRiskManager(make_config())
        big_order = make_order(size=2.0)
        approved = rm.pre_flight_check(
            "test", [big_order], InventoryState(), current_volatility=0.001,
            market_regime=MarketRegime.TRENDING,
        )
        assert approved == []
        events = rm.recent_events(10)
        assert any(e.event_type == "FAT_FINGER" for e in events)

    def test_rejects_order_priced_too_far_from_mid(self):
        rm = CentralRiskManager(make_config(max_price_deviation_pct=0.02))
        rm.update_market_state(mid=50_000.0)
        far_order = make_order(price=52_000.0, size=0.5)  # 4% away
        approved = rm.pre_flight_check(
            "test", [far_order], InventoryState(), current_volatility=0.001,
            market_regime=MarketRegime.TRENDING,
        )
        assert approved == []

    def test_accepts_reasonably_priced_small_order(self):
        rm = CentralRiskManager(make_config())
        rm.update_market_state(mid=50_000.0)
        ok_order = make_order(price=50_010.0, size=0.5)
        approved = rm.pre_flight_check(
            "test", [ok_order], InventoryState(), current_volatility=0.001,
            market_regime=MarketRegime.TRENDING,
        )
        assert len(approved) == 1
        assert approved[0].size == 0.5


class TestCooldownBlock:
    def test_report_toxic_fill_blocks_position_increasing_orders(self):
        rm = CentralRiskManager(make_config())
        rm.update_market_state(mid=50_000.0)
        rm.report_toxic_fill("test")
        assert rm.cooldown_remaining("test") > 0

        buy_order = make_order(side=Side.BUY, price=50_000.0, size=0.1)
        approved = rm.pre_flight_check(
            "test", [buy_order], InventoryState(net_position=0.0),
            current_volatility=0.001, market_regime=MarketRegime.TRENDING,
        )
        assert approved == []

    def test_cooldown_still_allows_orders_that_reduce_skew(self):
        rm = CentralRiskManager(make_config())
        rm.update_market_state(mid=50_000.0)
        rm.report_toxic_fill("test")
        # currently long 2.0 -> a SELL reduces skew, should be allowed through cooldown
        sell_order = make_order(side=Side.SELL, price=50_000.0, size=0.1)
        approved = rm.pre_flight_check(
            "test", [sell_order], InventoryState(net_position=2.0),
            current_volatility=0.001, market_regime=MarketRegime.TRENDING,
        )
        assert len(approved) == 1


class TestShaving:
    def test_shaves_order_down_to_available_headroom(self):
        rm = CentralRiskManager(make_config(max_net_position=5.0))
        rm.update_market_state(mid=50_000.0)
        # current_net=4.8, effective_max=5.0 -> headroom=0.2 for a BUY
        order = make_order(side=Side.BUY, price=50_000.0, size=1.0)
        approved = rm.pre_flight_check(
            "test", [order], InventoryState(net_position=4.8),
            current_volatility=0.001, market_regime=MarketRegime.TRENDING,
        )
        assert len(approved) == 1
        assert approved[0].size == pytest.approx(0.2, abs=1e-4)
        assert approved[0].size < order.size  # confirm it's actually the same order, shaved

    def test_discards_order_with_zero_headroom(self):
        rm = CentralRiskManager(make_config(max_net_position=5.0))
        rm.update_market_state(mid=50_000.0)
        order = make_order(side=Side.BUY, price=50_000.0, size=1.0)
        approved = rm.pre_flight_check(
            "test", [order], InventoryState(net_position=5.0),  # already at max
            current_volatility=0.001, market_regime=MarketRegime.TRENDING,
        )
        assert approved == []

    def test_always_allows_full_unwind_regardless_of_position_cap(self):
        rm = CentralRiskManager(make_config(max_net_position=5.0))
        rm.update_market_state(mid=50_000.0)
        # net=8.0 is already over max_net_position (shouldn't happen normally,
        # but a SELL that reduces skew must never be shaved down by the
        # position cap). Size kept under max_single_order_fraction so this
        # isolates the _shave stage rather than the separate FAT_FINGER
        # single-order-size check, which applies unconditionally regardless
        # of direction.
        order = make_order(side=Side.SELL, price=50_000.0, size=1.0)
        approved = rm.pre_flight_check(
            "test", [order], InventoryState(net_position=8.0),
            current_volatility=0.001, market_regime=MarketRegime.TRENDING,
        )
        assert len(approved) == 1
        assert approved[0].size == 1.0  # not shaved at all


class TestRegimeHaircuts:
    def test_toxic_regime_halves_effective_max_by_default(self):
        rm = CentralRiskManager(make_config(max_net_position=10.0, toxic_regime_haircut=0.5))
        normal = rm._effective_max_position(vol=0.0, regime=MarketRegime.TRENDING)
        toxic  = rm._effective_max_position(vol=0.0, regime=MarketRegime.TOXIC)
        assert toxic == pytest.approx(normal * 0.5)

    def test_illiquid_regime_applies_its_own_haircut(self):
        rm = CentralRiskManager(make_config(max_net_position=10.0, illiquid_regime_haircut=0.3))
        normal   = rm._effective_max_position(vol=0.0, regime=MarketRegime.TRENDING)
        illiquid = rm._effective_max_position(vol=0.0, regime=MarketRegime.ILLIQUID)
        assert illiquid == pytest.approx(normal * 0.7)

    def test_higher_volatility_shrinks_effective_max(self):
        rm = CentralRiskManager(make_config(max_net_position=10.0, vol_baseline=0.001))
        low_vol  = rm._effective_max_position(vol=0.001, regime=MarketRegime.TRENDING)
        high_vol = rm._effective_max_position(vol=0.01, regime=MarketRegime.TRENDING)
        assert high_vol < low_vol


class TestRecentEvents:
    """Regression test: recent_events() used to crash on every call because
    collections.deque doesn't support slice indexing (self._event_log[-n:]),
    unlike list. Found by actually calling this method while verifying the
    runner.py risk-manager wiring, not by inspection."""

    def test_does_not_raise_and_returns_events_newest_first(self):
        rm = CentralRiskManager(make_config())
        rm.report_toxic_fill("alpha")
        time.sleep(0.001)
        rm.report_toxic_fill("beta")

        events = rm.recent_events(10)  # must not raise TypeError
        assert len(events) == 2
        assert events[0].strategy == "beta"   # newest first
        assert events[1].strategy == "alpha"

    def test_respects_n_limit(self):
        rm = CentralRiskManager(make_config())
        for i in range(5):
            rm.report_toxic_fill(f"strat_{i}")
        events = rm.recent_events(2)
        assert len(events) == 2

    def test_empty_log_returns_empty_list(self):
        rm = CentralRiskManager(make_config())
        assert rm.recent_events(10) == []
