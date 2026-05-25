from __future__ import annotations

import copy
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from init import ExecutionOrder, InventoryState, Side, MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class RiskConfig:
    # Hard position and loss limits
    max_net_position: float
    max_drawdown_limit: float
    daily_loss_limit: float

    # Dynamic sizing scalars
    volatility_scalar: float = 1.0          # divisor applied when vol > vol_baseline
    vol_baseline: float = 0.001             # realized vol reference level (calibrate per venue)
    toxicity_cooldown_s: float = 60.0

    # Regime-based position haircuts (multiplicative, stacked on vol scaling)
    toxic_regime_haircut: float = 0.50      # halve max position in TOXIC regime
    illiquid_regime_haircut: float = 0.30

    # Fat-finger guards
    max_single_order_fraction: float = 0.25 # order.size ≤ max_net_position × this
    max_price_deviation_pct: float = 0.02   # reject if price deviates > 2% from last mid

    # Asset lot-size constraints (venue-specific — query from exchange meta on startup)
    size_decimals: int = 4                  # BTC perp on Hyperliquid = 4 (step = 0.0001)
    event_log_maxlen: int = 2_000           # rolling audit deque cap

@dataclass
class RiskEvent:
    ts: float
    event_type: str          # HALT | SHAVE | FAT_FINGER | COOLDOWN_BLOCK | TOXIC_FILL | UNHALT
    strategy: Optional[str]
    detail: str

class CentralRiskManager:

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

        self._min_size_step: float = math.pow(10.0, -config.size_decimals)

        self.is_halted: bool = False
        self._halt_reason: Optional[str] = None

        self._hwm: float = 0.0
        self._session_pnl: float = 0.0
        self._session_start_ts: float = time.time()

        self._cooldowns: Dict[str, float] = {}

        self._last_mid: Optional[float] = None

        self._event_log: Deque[RiskEvent] = deque(maxlen=config.event_log_maxlen)

    def update_market_state(self, mid: float) -> None:
        self._last_mid = mid

    def update_global_pnl(self, realized_pnl: float, unrealized_pnl: float) -> bool:
        if self.is_halted:
            return False

        self._session_pnl = realized_pnl + unrealized_pnl
        self._hwm         = max(self._hwm, self._session_pnl)
        drawdown          = self._hwm - self._session_pnl

        if drawdown > self.config.max_drawdown_limit:
            self._trigger_halt(
                f"Max drawdown breached: drawdown={drawdown:.4f} "
                f"limit={self.config.max_drawdown_limit:.4f} hwm={self._hwm:.4f}"
            )
            return False

        if self._session_pnl < -self.config.daily_loss_limit:
            self._trigger_halt(
                f"Daily loss limit breached: session_pnl={self._session_pnl:.4f} "
                f"limit={-self.config.daily_loss_limit:.4f}"
            )
            return False

        return True

    def pre_flight_check(
        self,
        strategy_name: str,
        proposed_orders: List[ExecutionOrder],
        current_inventory: InventoryState,
        current_volatility: float,
        market_regime: MarketRegime,
    ) -> List[ExecutionOrder]:
        if self.is_halted:
            return []

        effective_max = self._effective_max_position(current_volatility, market_regime)
        in_cooldown   = self._is_in_cooldown(strategy_name)
        running_net   = current_inventory.net_position

        approved: List[ExecutionOrder] = []

        for order in proposed_orders:
            result = self._evaluate_order(
                order, strategy_name, running_net, effective_max, in_cooldown
            )
            if result is not None:
                approved.append(result)
                delta = result.size if result.side == Side.BUY else -result.size
                running_net += delta

        return approved

    def _evaluate_order(
        self,
        order: ExecutionOrder,
        strategy_name: str,
        current_net: float,
        effective_max: float,
        in_cooldown: bool,
    ) -> Optional[ExecutionOrder]:
        max_single = self.config.max_net_position * self.config.max_single_order_fraction
        if order.size > max_single:
            self._log(
                "FAT_FINGER", strategy_name,
                f"size={order.size:.4f} > limit={max_single:.4f} [{order.side.value}]"
            )
            logger.warning(
                "FAT FINGER [%s]: size=%.4f exceeds single-order limit=%.4f — discarded",
                strategy_name, order.size, max_single,
            )
            return None
        if self._last_mid is not None:
            dev = abs(order.price - self._last_mid) / self._last_mid
            if dev > self.config.max_price_deviation_pct:
                self._log(
                    "FAT_FINGER", strategy_name,
                    f"price={order.price:.2f} dev={dev:.2%} > limit={self.config.max_price_deviation_pct:.2%} "
                    f"mid={self._last_mid:.2f}"
                )
                logger.warning(
                    "FAT FINGER [%s]: price %.2f deviates %.1f%% from mid %.2f — discarded",
                    strategy_name, order.price, dev * 100.0, self._last_mid,
                )
                return None
        if in_cooldown and not self._reduces_skew(order, current_net):
            remaining = self.cooldown_remaining(strategy_name)
            self._log(
                "COOLDOWN_BLOCK", strategy_name,
                f"{order.side.value} blocked, cooldown_remaining={remaining:.1f}s"
            )
            logger.info(
                "COOLDOWN [%s]: blocking %s — not a reduce, %.1fs remaining",
                strategy_name, order.side.value, remaining,
            )
            return None
        return self._shave(order, strategy_name, current_net, effective_max)

    def _shave(
        self,
        order: ExecutionOrder,
        strategy_name: str,
        current_net: float,
        effective_max: float,
    ) -> Optional[ExecutionOrder]:
        signed_delta = order.size if order.side == Side.BUY else -order.size
        new_net      = current_net + signed_delta

        if abs(new_net) <= effective_max:
            return order  # no shave needed

        if self._reduces_skew(order, current_net):
            return order  # always allow full unwind

        # Compute available headroom in the order direction
        headroom = (effective_max - current_net) if order.side == Side.BUY \
                   else (current_net + effective_max)

        if headroom < 1e-8:
            self._log(
                "SHAVE", strategy_name,
                f"zero headroom for {order.side.value}: "
                f"net={current_net:.4f} effective_max={effective_max:.4f} — discarded"
            )
            logger.info(
                "SHAVE [%s]: no headroom for %s (net=%.4f max=%.4f) — discarded",
                strategy_name, order.side.value, current_net, effective_max,
            )
            return None

        floored_size = math.floor(headroom / self._min_size_step) * self._min_size_step

        if floored_size < self._min_size_step:
            self._log(
                "SHAVE", strategy_name,
                f"headroom={headroom:.8f} < min_step={self._min_size_step:.8f} — discarded",
            )
            logger.info(
                "SHAVE [%s]: headroom %.8f below min lot step %.8f — discarded",
                strategy_name, headroom, self._min_size_step,
            )
            return None

        shaved      = copy.copy(order)
        shaved.size = round(floored_size, self.config.size_decimals)
        self._log(
            "SHAVE", strategy_name,
            f"{order.side.value} {order.size:.{self.config.size_decimals}f} "
            f"→ {shaved.size:.{self.config.size_decimals}f} "
            f"(net={current_net:.4f} max={effective_max:.4f} step={self._min_size_step})",
        )
        logger.info(
            "SHAVE [%s]: %s %.4f → %.4f (effective_max=%.4f)",
            strategy_name, order.side.value, order.size, shaved.size, effective_max,
        )
        return shaved

    def report_toxic_fill(self, strategy_name: str) -> None:
        now    = time.time()
        prev   = self._cooldowns.get(strategy_name, now)
        expiry = max(now, prev) + self.config.toxicity_cooldown_s
        max_expiry = now + self.config.toxicity_cooldown_s * 2.0
        self._cooldowns[strategy_name] = min(expiry, max_expiry)

        remaining = self._cooldowns[strategy_name] - now
        self._log("TOXIC_FILL", strategy_name, f"cooldown set, expires in {remaining:.1f}s")
        logger.warning("TOXIC FILL [%s]: cooldown %.1fs", strategy_name, remaining)

    def _is_in_cooldown(self, strategy_name: str) -> bool:
        return time.time() < self._cooldowns.get(strategy_name, 0.0)

    def cooldown_remaining(self, strategy_name: str) -> float:
        """Seconds remaining in cooldown. 0.0 if not active."""
        return max(0.0, self._cooldowns.get(strategy_name, 0.0) - time.time())

    def _effective_max_position(self, vol: float, regime: MarketRegime) -> float:
        vol_ratio  = vol / max(self.config.vol_baseline, 1e-9)
        vol_factor = max(1.0, vol_ratio * self.config.volatility_scalar)
        vol_scaled = self.config.max_net_position / vol_factor

        if regime == MarketRegime.TOXIC:
            return vol_scaled * (1.0 - self.config.toxic_regime_haircut)
        if regime == MarketRegime.ILLIQUID:
            return vol_scaled * (1.0 - self.config.illiquid_regime_haircut)

        return vol_scaled

    @staticmethod
    def _reduces_skew(order: ExecutionOrder, current_net: float) -> bool:
        """True if the order moves net position toward zero."""
        return (order.side == Side.SELL and current_net > 0) or \
               (order.side == Side.BUY  and current_net < 0)

    def _trigger_halt(self, reason: str) -> None:
        self.is_halted    = True
        self._halt_reason = reason
        self._log("HALT", None, reason)
        logger.critical("RISK HALT: %s", reason)

    def _log(self, event_type: str, strategy: Optional[str], detail: str) -> None:
        self._event_log.append(RiskEvent(
            ts=time.time(), event_type=event_type, strategy=strategy, detail=detail,
        ))

    @property
    def session_pnl(self) -> float:
        return self._session_pnl

    @property
    def drawdown(self) -> float:
        return self._hwm - self._session_pnl

    @property
    def halt_reason(self) -> Optional[str]:
        return self._halt_reason

    def status(self) -> dict:
        return {
            "is_halted":   self.is_halted,
            "halt_reason": self._halt_reason,
            "session_pnl": round(self._session_pnl, 6),
            "hwm":         round(self._hwm, 6),
            "drawdown":    round(self.drawdown, 6),
            "last_mid":    self._last_mid,
            "cooldowns":   {s: round(self.cooldown_remaining(s), 1) for s in self._cooldowns},
            "event_count": len(self._event_log),
        }

    def recent_events(self, n: int = 20) -> List[RiskEvent]:
        """Last N audit events, newest first."""
        return list(reversed(self._event_log[-n:]))

    def reset_session(self) -> None:
        self._session_pnl      = 0.0
        self._hwm              = 0.0
        self._cooldowns.clear()
        self._session_start_ts = time.time()
        self._log("SESSION_RESET", None, "PnL counters and cooldowns cleared")
        logger.info("Risk session reset.")

    def unhalt(self, operator_note: str = "") -> None:
        prev_reason = self._halt_reason
        self.is_halted    = False
        self._halt_reason = None
        self._log("UNHALT", None, f"previous_reason='{prev_reason}' note='{operator_note}'")
        logger.warning("UNHALT by operator. Was: '%s'. Note: '%s'", prev_reason, operator_note)
