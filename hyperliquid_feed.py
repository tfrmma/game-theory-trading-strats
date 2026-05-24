from __future__ import annotations

import asyncio
import logging
from queue import Queue
from typing import Callable, Dict, Any, List
import time

from hyperliquid.utils import constants
from hyperliquid.info import Info

from init import OrderBook, BookLevel, Trade, Side

logger = logging.getLogger(__name__)


class HyperliquidFeed:
    """Live WebSocket feed — converts raw Hyperliquid data to internal types."""

    def __init__(self, coin: str = "BTC", testnet: bool = False) -> None:
        self.coin    = coin.upper()
        self.testnet = testnet

        api_url   = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(api_url, skip_ws=False)

        self._book_queue:  Queue = Queue()
        self._trade_queue: Queue = Queue()
        self._running = False

    async def start(self, book_handler: Callable, trade_handler: Callable) -> None:
        self._book_handler  = book_handler
        self._trade_handler = trade_handler
        self._running       = True

        self.info.subscribe({"type": "l2Book", "coin": self.coin}, self._on_book_raw)
        self.info.subscribe({"type": "trades",  "coin": self.coin}, self._on_trade_raw)

        logger.info("Hyperliquid LIVE → %s (testnet=%s)", self.coin, self.testnet)

        asyncio.create_task(self._book_consumer())
        asyncio.create_task(self._trade_consumer())

    def _on_book_raw(self, data: Dict[str, Any]) -> None:
        if data.get("channel") == "l2Book" and data.get("data"):
            self._book_queue.put(data["data"])

    def _on_trade_raw(self, data: Dict[str, Any]) -> None:
        if data.get("channel") == "trades" and data.get("data"):
            self._trade_queue.put(data["data"])

    async def _book_consumer(self) -> None:
        while self._running:
            try:
                raw = await asyncio.get_running_loop().run_in_executor(None, self._book_queue.get)
                book = self._parse_l2book(raw)
                if book and self._book_handler:
                    await self._book_handler(book)
            except Exception as e:
                if self._running:
                    logger.error("Book consumer error: %s", e)

    async def _trade_consumer(self) -> None:
        while self._running:
            try:
                raw    = await asyncio.get_running_loop().run_in_executor(None, self._trade_queue.get)
                trades = self._parse_trades(raw)
                if trades and self._trade_handler:
                    await self._trade_handler(trades)
            except Exception as e:
                if self._running:
                    logger.error("Trade consumer error: %s", e)

    def _parse_l2book(self, data: Dict) -> Optional[OrderBook]:
        try:
            levels    = data.get("levels", [[], []])
            bids_raw, asks_raw = levels[0], levels[1]

            bids = [BookLevel(price=float(l["px"]), size=float(l["sz"]), order_count=1) for l in bids_raw[:20]]
            asks = [BookLevel(price=float(l["px"]), size=float(l["sz"]), order_count=1) for l in asks_raw[:20]]

            return OrderBook(bids=bids, asks=asks, timestamp=time.time(), sequence_id=int(data.get("ts", 0)))
        except Exception as e:
            logger.error("l2Book parse error: %s", e)
            return None

    def _parse_trades(self, data: List[Dict]) -> List[Trade]:
        trades = []
        for t in data:
            try:
                trades.append(Trade(
                    price=float(t["px"]),
                    size=float(t["sz"]),
                    side=Side.BUY if t.get("side") == "B" else Side.SELL,
                    timestamp=float(t.get("ts", time.time())),
                    is_liquidation=bool(t.get("liq", False)),
                    trade_id=str(t.get("tid")),
                ))
            except Exception:
                continue
        return trades

    async def stop(self) -> None:
        self._running = False
        logger.info("Hyperliquid feed stopped.")