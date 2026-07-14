"""
Tests for init.py's shared data structures: InventoryState (fill accounting +
unrealized_pnl), Side, and the two synthetic-market-data generators used
throughout the runner and every strategy's own demo function.
"""
from __future__ import annotations

import numpy as np
import pytest

from init import (
    InventoryState, Side, simulate_order_book, simulate_trade_tape,
    compute_realized_volatility,
)


class TestInventoryStateFillAccounting:
    def test_first_fill_sets_avg_entry_price(self):
        inv = InventoryState()
        inv.update_on_fill(Side.BUY, size=1.0, fill_price=100.0, mark_price=100.0, is_maker=True)
        assert inv.net_position == 1.0
        assert inv.avg_entry_price == 100.0
        assert inv.trade_count == 1

    def test_adding_to_same_side_weight_averages_entry_price(self):
        inv = InventoryState()
        inv.update_on_fill(Side.BUY, size=1.0, fill_price=100.0, mark_price=100.0, is_maker=True)
        inv.update_on_fill(Side.BUY, size=1.0, fill_price=110.0, mark_price=110.0, is_maker=True)
        assert inv.net_position == 2.0
        assert inv.avg_entry_price == pytest.approx(105.0)

    def test_closing_fill_realizes_pnl_at_avg_entry(self):
        inv = InventoryState()
        inv.update_on_fill(Side.BUY, size=2.0, fill_price=100.0, mark_price=100.0, is_maker=True)
        inv.update_on_fill(Side.SELL, size=1.0, fill_price=110.0, mark_price=110.0, is_maker=True)
        assert inv.realized_pnl == pytest.approx(10.0)  # 1.0 * (110 - 100)
        assert inv.net_position == pytest.approx(1.0)
        # avg_entry_price is untouched by a partial close (only updated on same-side adds)
        assert inv.avg_entry_price == pytest.approx(100.0)

    def test_short_side_realized_pnl_sign(self):
        inv = InventoryState()
        inv.update_on_fill(Side.SELL, size=1.0, fill_price=100.0, mark_price=100.0, is_maker=True)
        inv.update_on_fill(Side.BUY, size=1.0, fill_price=90.0, mark_price=90.0, is_maker=True)
        # short entered at 100, covered at 90 -> profit of 10
        assert inv.realized_pnl == pytest.approx(10.0)

    def test_maker_fill_accrues_adverse_selection_cost(self):
        inv = InventoryState()
        inv.update_on_fill(Side.BUY, size=1.0, fill_price=100.0, mark_price=100.5, is_maker=True)
        assert inv.adverse_selection_cost == pytest.approx(0.5)

    def test_taker_fill_does_not_accrue_adverse_selection_cost(self):
        inv = InventoryState()
        inv.update_on_fill(Side.BUY, size=1.0, fill_price=100.0, mark_price=100.5, is_maker=False)
        assert inv.adverse_selection_cost == 0.0


class TestUnrealizedPnl:
    """Regression tests for the unrealized_pnl fix — this property used to be
    a stub that always returned 0.0 regardless of position or mark price."""

    def test_flat_position_has_zero_unrealized_pnl(self):
        inv = InventoryState()
        assert inv.unrealized_pnl == 0.0

    def test_no_fills_yet_has_zero_unrealized_pnl(self):
        inv = InventoryState(net_position=5.0, avg_entry_price=100.0)  # constructed directly, no fill
        assert inv.unrealized_pnl == 0.0  # last_mark_price is still 0.0

    def test_long_position_marks_to_market_correctly(self):
        inv = InventoryState()
        inv.update_on_fill(Side.BUY, size=2.0, fill_price=100.0, mark_price=100.0, is_maker=True)
        inv.last_mark_price = 110.0  # simulate a later mark update without a new fill
        assert inv.unrealized_pnl == pytest.approx(20.0)  # 2.0 * (110 - 100)

    def test_short_position_marks_to_market_correctly(self):
        inv = InventoryState()
        inv.update_on_fill(Side.SELL, size=2.0, fill_price=100.0, mark_price=100.0, is_maker=True)
        inv.last_mark_price = 90.0
        assert inv.unrealized_pnl == pytest.approx(20.0)  # -2.0 * (90 - 100)

    def test_short_position_underwater_is_negative(self):
        inv = InventoryState()
        inv.update_on_fill(Side.SELL, size=1.0, fill_price=100.0, mark_price=100.0, is_maker=True)
        inv.last_mark_price = 110.0
        assert inv.unrealized_pnl == pytest.approx(-10.0)

    def test_last_mark_price_updates_on_every_fill(self):
        inv = InventoryState()
        inv.update_on_fill(Side.BUY, size=1.0, fill_price=100.0, mark_price=99.0, is_maker=True)
        assert inv.last_mark_price == 99.0
        inv.update_on_fill(Side.BUY, size=1.0, fill_price=101.0, mark_price=103.0, is_maker=True)
        assert inv.last_mark_price == 103.0


class TestSide:
    def test_opposite(self):
        assert Side.BUY.opposite() == Side.SELL
        assert Side.SELL.opposite() == Side.BUY


class TestSyntheticMarketData:
    """These generators feed every strategy's own demo function and the
    runner's simulation loop — a regression here would silently break every
    other test and every demo in the repo."""

    def test_simulate_order_book_bids_below_asks(self):
        book = simulate_order_book(mid=50_000.0)
        assert len(book.bids) > 0 and len(book.asks) > 0
        assert book.bids[0].price < book.asks[0].price
        assert book.mid == pytest.approx(50_000.0, abs=book.asks[0].price - book.bids[0].price)

    def test_simulate_order_book_levels_sorted_correctly(self):
        book = simulate_order_book(mid=50_000.0)
        bid_prices = [l.price for l in book.bids]
        ask_prices = [l.price for l in book.asks]
        assert bid_prices == sorted(bid_prices, reverse=True)  # best bid first, descending
        assert ask_prices == sorted(ask_prices)                # best ask first, ascending

    def test_simulate_trade_tape_returns_requested_count(self):
        rng = np.random.default_rng(0)
        trades = simulate_trade_tape(n_trades=25, mid=50_000.0, informed_fraction=0.1, rng=rng)
        assert len(trades) == 25
        assert all(t.price > 0 and t.size > 0 for t in trades)

    def test_simulate_trade_tape_is_deterministic_given_seed(self):
        trades_a = simulate_trade_tape(10, 50_000.0, 0.1, rng=np.random.default_rng(7))
        trades_b = simulate_trade_tape(10, 50_000.0, 0.1, rng=np.random.default_rng(7))
        assert [t.price for t in trades_a] == [t.price for t in trades_b]


class TestRealizedVolatility:
    def test_returns_zero_for_insufficient_data(self):
        assert compute_realized_volatility(np.array([100.0])) == 0.0

    def test_returns_zero_for_constant_prices(self):
        prices = np.full(50, 100.0)
        assert compute_realized_volatility(prices) == pytest.approx(0.0, abs=1e-9)

    def test_positive_for_moving_prices(self):
        rng = np.random.default_rng(1)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
        assert compute_realized_volatility(prices) > 0.0
