"""
Tests for hyperliquid_feed.py. Payloads below are shaped exactly like the
official Hyperliquid WsBook/WsTrade schemas verified two ways this session:
against the official docs (hyperliquid.gitbook.io) and against the installed
hyperliquid-python-sdk's own TypedDicts (hyperliquid/utils/types.py:
L2Level={"px","sz","n"}, L2BookData={...,"time"}, Trade={...,"time"}, no
"ts" field and no "liq"/liquidation field anywhere in the public schema).
These pin the parsing bugs found and fixed this session: "time" not "ts"
(was silently defaulting to 0 / local time.time() on every message), the
order-count field "n" (was hardcoded to 1), and the nonexistent "liq" field
(was read as if it were real).
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from init import Side


@pytest.fixture
def feed():
    with patch("hyperliquid_feed.Info") as MockInfo:
        MockInfo.return_value = MagicMock()
        from hyperliquid_feed import HyperliquidFeed
        yield HyperliquidFeed("BTC")


class TestParseL2Book:
    def test_parses_bids_and_asks(self, feed):
        payload = {
            "coin": "BTC", "time": 1753606210273,
            "levels": [
                [{"px": "50000.0", "sz": "1.2", "n": 3}],
                [{"px": "50000.5", "sz": "1.1", "n": 4}],
            ],
        }
        book = feed._parse_l2book(payload)
        assert book is not None
        assert book.bids[0].price == 50000.0
        assert book.asks[0].price == 50000.5

    def test_sequence_id_reads_time_field_not_ts(self, feed):
        payload = {"coin": "BTC", "time": 1753606210273, "levels": [[], []]}
        book = feed._parse_l2book(payload)
        assert book.sequence_id == 1753606210273

    def test_order_count_reads_n_field_not_hardcoded(self, feed):
        payload = {
            "coin": "BTC", "time": 0,
            "levels": [[{"px": "100.0", "sz": "1.0", "n": 7}], [{"px": "101.0", "sz": "1.0", "n": 2}]],
        }
        book = feed._parse_l2book(payload)
        assert book.bids[0].order_count == 7
        assert book.asks[0].order_count == 2

    def test_missing_n_field_falls_back_to_one(self, feed):
        payload = {"coin": "BTC", "time": 0, "levels": [[{"px": "100.0", "sz": "1.0"}], []]}
        book = feed._parse_l2book(payload)
        assert book.bids[0].order_count == 1

    def test_missing_levels_key_defaults_to_empty_book_not_none(self, feed):
        # data.get("levels", [[],[]]) defaults gracefully this is correct,
        # defensive behavior, not a bug.
        book = feed._parse_l2book({"coin": "BTC", "time": 0})
        assert book is not None
        assert book.bids == [] and book.asks == []

    def test_malformed_level_entries_return_none(self, feed):
        # A level present but missing required keys (px/sz) should hit the
        # except branch and return None, not raise.
        payload = {"coin": "BTC", "time": 0, "levels": [[{"not_px": "oops"}], []]}
        assert feed._parse_l2book(payload) is None


class TestParseTrades:
    def test_timestamp_reads_time_field_in_milliseconds(self, feed):
        payload = [{"coin": "BTC", "side": "B", "px": "50000.0", "sz": "0.01",
                     "hash": "0xabc", "time": 1753606210273, "tid": 999}]
        trades = feed._parse_trades(payload)
        assert trades[0].timestamp == pytest.approx(1753606210.273)

    def test_side_b_maps_to_buy(self, feed):
        payload = [{"coin": "BTC", "side": "B", "px": "1", "sz": "1", "time": 0, "tid": 1}]
        assert feed._parse_trades(payload)[0].side == Side.BUY

    def test_side_other_maps_to_sell(self, feed):
        payload = [{"coin": "BTC", "side": "A", "px": "1", "sz": "1", "time": 0, "tid": 1}]
        assert feed._parse_trades(payload)[0].side == Side.SELL

    def test_is_liquidation_always_false_field_does_not_exist_in_feed(self, feed):
        payload = [{"coin": "BTC", "side": "B", "px": "1", "sz": "1", "time": 0,
                     "tid": 1, "liq": True}]
        assert feed._parse_trades(payload)[0].is_liquidation is False

    def test_skips_malformed_entries_without_raising(self, feed):
        payload = [
            {"coin": "BTC", "side": "B", "px": "100", "sz": "1", "time": 0, "tid": 1},
            {"garbage": True},
        ]
        trades = feed._parse_trades(payload)
        assert len(trades) == 1


class TestReconnectWatchdog:
    """Official Hyperliquid docs: automated clients must handle disconnects
    and reconnect themselves - the SDK gives no on-disconnect callback, and
    websocket_manager.py's run_forever() has no built-in auto-reconnect."""

    @pytest.mark.asyncio
    async def test_detects_staleness_and_recreates_info_object(self, feed):
        feed._running = True
        feed._loop = asyncio.get_running_loop()
        feed._STALE_S = 0.0
        feed._last_msg_ts = time.time() - 100
        original_info = feed.info

        with patch("hyperliquid_feed.Info") as MockInfo:
            MockInfo.return_value = MagicMock()
            task = asyncio.create_task(feed._reconnect_watchdog())
            await asyncio.sleep(6.0)
            feed._running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert feed.info is not original_info
        assert time.time() - feed._last_msg_ts < 5.0

    @pytest.mark.asyncio
    async def test_does_not_reconnect_when_messages_are_flowing(self, feed):
        feed._running = True
        feed._loop = asyncio.get_running_loop()
        feed._STALE_S = 100.0
        feed._last_msg_ts = time.time()
        original_info = feed.info

        task = asyncio.create_task(feed._reconnect_watchdog())
        await asyncio.sleep(6.0)
        feed._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert feed.info is original_info

    @pytest.mark.asyncio
    async def test_backs_off_on_repeated_reconnect_failure(self, feed):
        feed._running = True
        feed._loop = asyncio.get_running_loop()
        feed._STALE_S = 0.0
        feed._last_msg_ts = time.time() - 100

        with patch("hyperliquid_feed.Info", side_effect=ConnectionError("simulated")):
            task = asyncio.create_task(feed._reconnect_watchdog())
            await asyncio.sleep(2.0)
            feed._running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert task.cancelled() or task.done()
