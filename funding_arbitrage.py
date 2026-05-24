from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from init import FundingSnapshot, Trade, OrderBook, Side, Signal, SignalStrength, simulate_order_book, simulate_trade_tape

logger = logging.getLogger(__name__)

# 8h epoch constants
FUNDING_INTERVAL_S        = 8 * 3600
EXTREME_FUNDING_THRESHOLD = 0.0005   # ~22% annualized
VERY_EXTREME_THRESHOLD    = 0.001    # ~45% annualized
PRE_PRINT_ENTRY_WINDOW_S  = 45 * 60
POST_PRINT_EXIT_WINDOW_S  = 5  * 60


class FundingMonitor:
    """
    Tracks funding rate history and flags extreme/unsustainable regimes.
    Z-score > 2σ from rolling mean = statistically extreme, mean-reversion likely within 1-3 epochs.
    """

    def __init__(self, lookback_epochs: int = 21, z_score_extreme: float = 2.0) -> None:
        self.lookback_epochs = lookback_epochs
        self.z_score_extreme = z_score_extreme
        self._rate_history:  Deque[float] = deque(maxlen=lookback_epochs)
        self._oi_history:    Deque[float] = deque(maxlen=lookback_epochs)
        self._basis_history: Deque[float] = deque(maxlen=lookback_epochs)
        self.latest: Optional[FundingSnapshot] = None

    def update(self, snapshot: FundingSnapshot) -> None:
        self.latest = snapshot
        self._rate_history.append(snapshot.rate)
        self._oi_history.append(snapshot.open_interest)
        self._basis_history.append(snapshot.basis_bps)

    @property
    def current_rate(self) -> float:
        return self.latest.rate if self.latest else 0.0

    @property
    def annualized_rate(self) -> float:
        return self.current_rate * 3 * 365

    @property
    def z_score(self) -> float:
        if len(self._rate_history) < 5:
            return 0.0
        rates = np.array(self._rate_history)
        mu    = float(rates[:-1].mean())
        sigma = float(rates[:-1].std()) + 1e-9
        return (rates[-1] - mu) / sigma

    @property
    def oi_trend(self) -> float:
        if len(self._oi_history) < 4:
            return 0.0
        recent = list(self._oi_history)
        return (recent[-1] - recent[-4]) / (recent[-4] + 1e-9)

    def is_extreme_positive(self) -> bool:
        return self.current_rate > EXTREME_FUNDING_THRESHOLD and self.z_score > self.z_score_extreme

    def is_extreme_negative(self) -> bool:
        return self.current_rate < -EXTREME_FUNDING_THRESHOLD and self.z_score < -self.z_score_extreme

    def regime_summary(self) -> str:
        if self.is_extreme_positive():
            return f"EXTREME_LONG_CROWDING (rate={self.current_rate:.4%} z={self.z_score:.1f})"
        if self.is_extreme_negative():
            return f"EXTREME_SHORT_CROWDING (rate={self.current_rate:.4%} z={self.z_score:.1f})"
        return f"NORMAL (rate={self.current_rate:.4%} z={self.z_score:.1f})"


class SpotPerpDivergenceDetector:
    """
    Flags divergence between spot CVD and perp price action.
    Perp up + spot CVD flat/down = distribution, not genuine demand. Core funding trade tell.
    """

    def __init__(self, cvd_window: int = 50) -> None:
        self._spot_cvd: float = 0.0
        self._perp_mid_history:  Deque[float] = deque(maxlen=cvd_window)
        self._spot_cvd_history:  Deque[float] = deque(maxlen=cvd_window)

    def update_spot_trades(self, trades: List[Trade]) -> None:
        for t in trades:
            self._spot_cvd += t.size if t.side == Side.BUY else -t.size
        self._spot_cvd_history.append(self._spot_cvd)

    def update_perp_mid(self, mid: float) -> None:
        self._perp_mid_history.append(mid)

    def is_diverging(self, lookback: int = 20) -> Tuple[bool, str]:
        if len(self._perp_mid_history) < lookback or len(self._spot_cvd_history) < lookback:
            return False, "Insufficient data"

        perp_mids = list(self._perp_mid_history)[-lookback:]
        spot_cvds = list(self._spot_cvd_history)[-lookback:]
        perp_up   = perp_mids[-1] > perp_mids[0]
        cvd_up    = spot_cvds[-1] > spot_cvds[0]

        if perp_up and not cvd_up:
            return True, f"DISTRIBUTION: perp up, spot CVD={spot_cvds[-1]-spot_cvds[0]:.1f}"
        if not perp_up and cvd_up:
            return True, f"ACCUMULATION: perp down, spot CVD=+{spot_cvds[-1]-spot_cvds[0]:.1f}"
        return False, "No divergence"


@dataclass
class FundingArbitragePosition:
    perp_side: Side
    perp_size: float
    perp_entry: float
    spot_size: float
    spot_entry: float
    entry_ts: float
    target_exit_ts: float
    epochs_held: int = 0
    funding_collected: float = 0.0

    @property
    def hold_duration_s(self) -> float:
        return time.time() - self.entry_ts

    def collect_funding(self, rate: float) -> float:
        notional = self.perp_size * self.perp_entry
        funding  = rate * notional if self.perp_side == Side.SELL else -rate * notional
        self.funding_collected += funding
        self.epochs_held += 1
        return funding


class FundingArbitrageStrategy:
    """
    Two modes:
      static   — delta-neutral carry, hold multiple epochs, exit when rate normalizes.
      tactical — enter 45min before print, collect one payment, unwind within 5min of print.
    """

    def __init__(
        self,
        mode: str = "tactical",
        max_position_notional: float = 100_000.0,
        entry_z_threshold: float = 2.0,
        spot_cvd_diverge_required: bool = True,
        entry_window_s: float = PRE_PRINT_ENTRY_WINDOW_S,
        exit_window_s: float = POST_PRINT_EXIT_WINDOW_S,
        rate_normalization_threshold: float = 0.0001,
    ) -> None:
        self.mode                         = mode
        self.max_position_notional        = max_position_notional
        self.entry_z_threshold            = entry_z_threshold
        self.spot_cvd_diverge_req         = spot_cvd_diverge_required
        self.entry_window_s               = entry_window_s
        self.exit_window_s                = exit_window_s
        self.rate_normalization_threshold = rate_normalization_threshold

        self.funding_monitor  = FundingMonitor()
        self.divergence_det   = SpotPerpDivergenceDetector()
        self._active_position: Optional[FundingArbitragePosition] = None
        self._pnl_log: List[dict] = []

    def update_funding(self, snapshot: FundingSnapshot, spot_trades: List[Trade]) -> None:
        self.funding_monitor.update(snapshot)
        self.divergence_det.update_spot_trades(spot_trades)
        self.divergence_det.update_perp_mid(snapshot.mark_price)

        if self._active_position:
            earned = self._active_position.collect_funding(snapshot.rate)
            logger.info("Funding collected: %.4f USD (total: %.4f USD)", earned, self._active_position.funding_collected)

    def evaluate_entry(self, snapshot: FundingSnapshot) -> Optional[Signal]:
        if self._active_position is not None:
            return None

        fund_mon    = self.funding_monitor
        extreme_pos = fund_mon.is_extreme_positive()
        extreme_neg = fund_mon.is_extreme_negative()

        if not (extreme_pos or extreme_neg):
            return None

        diverging, diverge_desc = self.divergence_det.is_diverging()
        if self.spot_cvd_diverge_req and not diverging:
            logger.debug("Funding extreme but no spot-perp divergence — skipping")
            return None

        if self.mode == "tactical":
            time_to_print = snapshot.seconds_to_print
            if not (0 < time_to_print <= self.entry_window_s):
                return None

        rate_extremity = min(1.0, abs(fund_mon.current_rate) / VERY_EXTREME_THRESHOLD)
        notional = self.max_position_notional * rate_extremity
        size     = notional / snapshot.mark_price

        if extreme_pos:
            perp_side   = Side.SELL
            target_exit = snapshot.next_funding_ts + self.exit_window_s
            rationale   = (
                f"FUNDING ARB (short perp): rate={fund_mon.current_rate:.4%} "
                f"z={fund_mon.z_score:.2f} | {diverge_desc} | "
                f"time_to_print={snapshot.seconds_to_print/60:.1f}min"
            )
        else:
            perp_side   = Side.BUY
            target_exit = snapshot.next_funding_ts + self.exit_window_s
            rationale   = (
                f"FUNDING ARB (long perp): rate={fund_mon.current_rate:.4%} "
                f"z={fund_mon.z_score:.2f} | {diverge_desc}"
            )

        self._active_position = FundingArbitragePosition(
            perp_side=perp_side,
            perp_size=size,
            perp_entry=snapshot.mark_price,
            spot_size=size,
            spot_entry=snapshot.index_price,
            entry_ts=time.time(),
            target_exit_ts=target_exit,
        )

        logger.info("FUNDING ARB ENTERED: %s", rationale)

        return Signal(
            direction=perp_side,
            strength=SignalStrength.STRONG if rate_extremity > 0.7 else SignalStrength.MEDIUM,
            confidence=min(1.0, rate_extremity * (1.2 if diverging else 0.8)),
            entry_price=snapshot.mark_price,
            target_price=snapshot.index_price,
            stop_price=snapshot.mark_price * (1.015 if perp_side == Side.SELL else 0.985),
            rationale=rationale,
            invalidation_conditions=[
                "Spot CVD confirms perp direction",
                "OI expands through funding print",
                "Rate increases further after entry",
                "Macro news within 30min of print",
            ],
        )

    def evaluate_exit(self, snapshot: FundingSnapshot) -> Optional[str]:
        if not self._active_position:
            return None

        pos = self._active_position
        now = time.time()

        if self.mode == "tactical" and now > pos.target_exit_ts:
            return f"Post-print exit. Funding collected: {pos.funding_collected:.4f}"

        if self.mode == "static":
            if abs(self.funding_monitor.current_rate) < self.rate_normalization_threshold:
                return f"Rate normalized to {self.funding_monitor.current_rate:.4%}"

        basis_bps = snapshot.basis_bps
        if pos.perp_side == Side.SELL and basis_bps > 50:
            return f"Stop: basis widened to {basis_bps:.1f}bps"
        if pos.perp_side == Side.BUY and basis_bps < -50:
            return f"Stop: basis narrowed to {basis_bps:.1f}bps"

        return None

    def close_position(self, snapshot: FundingSnapshot, reason: str) -> dict:
        if not self._active_position:
            return {}

        pos = self._active_position
        perp_pnl = (snapshot.mark_price - pos.perp_entry) * pos.perp_size * (-1 if pos.perp_side == Side.SELL else 1)
        spot_pnl = (snapshot.index_price - pos.spot_entry) * pos.spot_size

        pnl_summary = {
            "epochs_held":     pos.epochs_held,
            "funding_pnl":     pos.funding_collected,
            "perp_pnl":        perp_pnl,
            "spot_pnl":        spot_pnl,
            "total_pnl":       pos.funding_collected + perp_pnl + spot_pnl,
            "hold_duration_h": pos.hold_duration_s / 3600,
            "exit_reason":     reason,
        }

        self._pnl_log.append(pnl_summary)
        logger.info("FUNDING ARB CLOSED: %s", pnl_summary)
        self._active_position = None
        return pnl_summary


def simulate_funding_arb(n_epochs: int = 10, base_mid: float = 50_000.0) -> None:
    rng      = np.random.default_rng(42)
    strategy = FundingArbitrageStrategy(mode="tactical")

    print("\n" + "="*65)
    print("FUNDING RATE ARBITRAGE SIMULATION")
    print("="*65)

    index_price = base_mid * 0.998
    mark_price  = base_mid
    base_rate   = 0.0008

    for epoch in range(n_epochs):
        rate_decay  = 1 - (epoch / n_epochs) * 0.6
        rate        = base_rate * rate_decay * rng.lognormal(0, 0.15)
        mark_price  = index_price * (1 + 0.002 * (1 - epoch / n_epochs))
        next_print  = time.time() + 40 * 60

        snapshot = FundingSnapshot(
            rate=rate,
            next_funding_ts=next_print,
            open_interest=1_000_000 * rng.lognormal(0, 0.05),
            mark_price=mark_price,
            index_price=index_price,
            timestamp=time.time(),
        )

        # Bias spot CVD toward sell to simulate distribution tell
        spot_trades = simulate_trade_tape(20, index_price, 10.0, informed_fraction=0.05, rng=rng)
        for t in spot_trades:
            if rng.random() < 0.6:
                t.side = Side.SELL

        strategy.update_funding(snapshot, spot_trades)

        print(f"\nEpoch {epoch+1:2d} | rate={rate:.4%} | basis={snapshot.basis_bps:.1f}bps")
        print(f"  Regime: {strategy.funding_monitor.regime_summary()}")

        entry_signal = strategy.evaluate_entry(snapshot)
        if entry_signal:
            print(f"  ✓ ENTRY: {entry_signal.direction.value.upper()} perp | {entry_signal.rationale}")

        exit_reason = strategy.evaluate_exit(snapshot)
        if exit_reason:
            pnl = strategy.close_position(snapshot, exit_reason)
            print(f"  ✗ EXIT: {exit_reason}")
            print(f"    PnL: funding={pnl['funding_pnl']:.2f} perp={pnl['perp_pnl']:.2f} total={pnl['total_pnl']:.2f}")
            break

    print("\n" + "="*65 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    simulate_funding_arb()