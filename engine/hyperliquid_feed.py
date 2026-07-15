from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, Any, List, Optional
import time

from hyperliquid.utils import constants
from hyperliquid.info import Info

from engine.init import OrderBook, BookLevel, Trade, Side

logger = logging.getLogger(__name__)


class HyperliquidFeed:
    """Live WebSocket feed converts raw Hyperliquid data to internal types."""

    def __init__(self, coin: str = "BTC", testnet: bool = False) -> None:
        self.coin    = coin.upper()
        self.testnet = testnet

        api_url   = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(api_url, skip_ws=False)

        # asyncio.Queue owned by the event loop; SDK callbacks write via call_soon_threadsafe.
        self._book_queue:  asyncio.Queue = asyncio.Queue()
        self._trade_queue: asyncio.Queue = asyncio.Queue()
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._running = False

        # Reconnection: official docs say "all automated users should handle
        # disconnects from the server side and gracefully reconnect" the SDK
        # gives no on-disconnect callback, so staleness (no message in
        # STALE_S seconds) is the signal. l2Book sends a snapshot + deltas
        # continuously on any active market, so silence this long means the
        # connection is dead, not that the market is quiet.
        self._STALE_S        = 30.0
        self._MAX_BACKOFF_S  = 60.0
        self._last_msg_ts    = time.time()
        self._reconnect_task: Optional[asyncio.Task] = None

    def get_funding_ctx(self) -> Optional[Dict[str, float]]:
        """REST snapshot of this coin's current funding rate/mark/oracle/OI, via
        the official SDK's Info.meta_and_asset_ctxs() (POST /info, type
        metaAndAssetCtxs) self.info already exists for this, it just wasn't
        being called. Hyperliquid settles funding hourly and the rate is
        recomputed continuously from a 5s-sampled premium average (per docs),
        so this is a REST poll meant to be called periodically (e.g. once a
        minute), not per-tick it is not a WebSocket push.
        """
        try:
            meta, asset_ctxs = self.info.meta_and_asset_ctxs()
            for asset, ctx in zip(meta["universe"], asset_ctxs):
                if asset["name"] == self.coin:
                    return {
                        "rate":           float(ctx["funding"]),
                        "mark_price":     float(ctx["markPx"]),
                        "oracle_price":   float(ctx["oraclePx"]),
                        "open_interest":  float(ctx["openInterest"]),
                    }
            logger.warning("get_funding_ctx: %s not found in universe", self.coin)
            return None
        except Exception as e:
            logger.error("get_funding_ctx error: %s", e)
            return None

    async def start(self, book_handler: Callable, trade_handler: Callable) -> None:
        self._book_handler  = book_handler
        self._trade_handler = trade_handler
        self._loop          = asyncio.get_running_loop()
        self._running       = True

        self._subscribe()

        logger.info("Hyperliquid LIVE → %s (testnet=%s)", self.coin, self.testnet)

        asyncio.create_task(self._book_consumer())
        asyncio.create_task(self._trade_consumer())
        self._reconnect_task = asyncio.create_task(self._reconnect_watchdog())

    def _subscribe(self) -> None:
        self.info.subscribe({"type": "l2Book", "coin": self.coin}, self._on_book_raw)
        self.info.subscribe({"type": "trades",  "coin": self.coin}, self._on_trade_raw)

    #  SDK callbacks (background thread) 

    def _on_book_raw(self, data: Dict[str, Any]) -> None:
        self._last_msg_ts = time.time()
        if data.get("channel") == "l2Book" and data.get("data") and self._loop:
            self._loop.call_soon_threadsafe(self._book_queue.put_nowait, data["data"])

    def _on_trade_raw(self, data: Dict[str, Any]) -> None:
        self._last_msg_ts = time.time()
        if data.get("channel") == "trades" and data.get("data") and self._loop:
            self._loop.call_soon_threadsafe(self._trade_queue.put_nowait, data["data"])

    # Consumers (event loop) 

    async def _book_consumer(self) -> None:
        while self._running:
            try:
                raw  = await self._book_queue.get()
                book = self._parse_l2book(raw)
                if book:
                    await self._book_handler(book)
            except Exception as e:
                if self._running:
                    logger.error("Book consumer error: %s", e)

    async def _trade_consumer(self) -> None:
        while self._running:
            try:
                raw    = await self._trade_queue.get()
                trades = self._parse_trades(raw)
                if trades:
                    await self._trade_handler(trades)
            except Exception as e:
                if self._running:
                    logger.error("Trade consumer error: %s", e)

    # Parsers 

    def _parse_l2book(self, data: Dict) -> Optional[OrderBook]:
        # WsBook schema (official docs): { coin, levels: [WsLevel[], WsLevel[]], time }
        # WsLevel: { px, sz, n }. There is no "ts" field that was silently
        # falling back to 0 every message. "n" (order count at the level) was
        # being discarded and hardcoded to 1, even though BookLevel.order_count
        # exists specifically to carry it (used by queue_warfare.py's alpha
        # cancellation model).
        try:
            levels           = data.get("levels", [[], []])
            bids_raw, asks_raw = levels[0], levels[1]
            bids = [
                BookLevel(price=float(l["px"]), size=float(l["sz"]), order_count=int(l.get("n", 1)))
                for l in bids_raw[:20]
            ]
            asks = [
                BookLevel(price=float(l["px"]), size=float(l["sz"]), order_count=int(l.get("n", 1)))
                for l in asks_raw[:20]
            ]
            return OrderBook(bids=bids, asks=asks, timestamp=time.time(), sequence_id=int(data.get("time", 0)))
        except Exception as e:
            logger.error("l2Book parse error: %s", e)
            return None

    def _parse_trades(self, data: List[Dict]) -> List[Trade]:
        # WsTrade schema (official docs): { coin, side, px, sz, hash, time, tid, users }
        # No "ts" field (was silently falling back to time.time() local receipt
        # time, not exchange trade time on every single trade) and no "liq"
        # field: the public trades feed carries no per-trade liquidation flag.
        # Liquidation info only exists on WsUserEvent/WsUserNonFundingLedgerUpdates,
        # which are per-user authenticated streams, not public market data.
        # is_liquidation is left False here rather than reading a field that
        # never exists in this payload. (Note: liquidation_frontrun.py's
        # CascadeDetector doesn't actually key off is_liquidation it infers
        # cascades from volume spike + OI drop so this doesn't silently break
        # that strategy, but the field was dead/misleading as written.)
        trades = []
        for t in data:
            try:
                trades.append(Trade(
                    price=float(t["px"]),
                    size=float(t["sz"]),
                    side=Side.BUY if t.get("side") == "B" else Side.SELL,
                    timestamp=float(t.get("time", time.time())) / 1000.0,
                    is_liquidation=False,
                    trade_id=str(t.get("tid")),
                ))
            except Exception:
                continue
        return trades

    async def stop(self) -> None:
        self._running = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
        logger.info("Hyperliquid feed stopped.")

    async def _reconnect_watchdog(self) -> None:
        """No message (book or trade) for _STALE_S seconds -> assume the
        connection is dead and reconnect. Backoff doubles each consecutive
        failed attempt, capped at _MAX_BACKOFF_S, and resets to 1s once a
        message is received again."""
        backoff = 1.0
        while self._running:
            await asyncio.sleep(5.0)
            if not self._running:
                return
            idle = time.time() - self._last_msg_ts
            if idle <= self._STALE_S:
                backoff = 1.0
                continue

            logger.warning(
                "Hyperliquid feed stale (%.0fs since last message) reconnecting (backoff=%.0fs)",
                idle, backoff,
            )
            try:
                api_url   = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
                self.info = Info(api_url, skip_ws=False)
                self._subscribe()
                self._last_msg_ts = time.time()
                logger.info("Hyperliquid feed reconnected → %s", self.coin)
                backoff = 1.0
            except Exception as e:
                logger.error("Reconnect attempt failed: %s", e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._MAX_BACKOFF_S)
