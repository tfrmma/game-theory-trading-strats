# game-theory-trading-strats

Game-theoretic microstructure strategies for crypto perpetuals.
Tested on Hyperliquid; adaptable to Binance Perps, dYdX, and similar venues.

---

## Modules

### Strategies

| Module | Game | Edge |
|--------|------|------|
| `spoofing_counter.py` | Crawford-Sobel Signaling | Detect fake depth, fade the illusion |
| `predatory_liquidity.py` | Stackelberg Coordination | Join or fade stop cascades |
| `info_asymmetry.py` | Glosten-Milgrom Adverse Selection | VPIN + Rolling ADV, Kyle's λ, flow toxicity |
| `queue_warfare.py` | FIFO Queue Leadership | Iceberg detection, stochastic cancel factor α, OFI-triggered cancel |
| `funding_arbitrage.py` | Convergence Timing Game | Pre-print funding capture, spot-perp divergence |
| `liquidation_frontrun.py` | Dominated Strategy Exploitation | Front-run deterministic liq engines |
| `adaptive_guerrilla.py` | Avellaneda-Stoikov + toxic flow cancel | Decaying risk horizon T-t, inventory skew, adverse selection deflection |

### Infrastructure

| Module | Role |
|--------|------|
| `init.py` | Shared data structures, enums, simulation utilities |
| `runner.py` | Central orchestrator — simulation and live Hyperliquid modes |
| `hyperliquid_feed.py` | Live WebSocket feed with `asyncio.Queue` + `call_soon_threadsafe` |
| `central_risk_manager.py` | Pre-trade risk gate: circuit breaker, sizing shaver, fat-finger, toxicity cooldown |
| `backtester.py` | Tick-by-tick historical replay, passive fill simulation, PnL attribution |
| `adverse_selection.py` | Markout PnL, effective/realized spread decomposition, Roll estimator, Amihud ratio |
| `hot_paths.py` | Auto-selecting wrapper — Cython extension or pure Python fallback |
| `_hot_paths.pyx` | Cython hot paths: OFI, VPIN bucket fill, Kyle's λ OLS, Poisson fill prob, lot floor |
| `_hot_paths_pure.py` | Pure Python fallback — identical API to the compiled extension |
| `setup_hot_paths.py` | Build script for the Cython extension |

---

## Setup

```bash
pip install -r requirements.txt

# Optional: compile Cython hot paths (~10-100x speedup on inner loops)
python setup_hot_paths.py build_ext --inplace
```

---

## Run

```bash
# Simulation
python runner.py

# Live (real money)
python runner.py --live --coin BTC

# Testnet
python runner.py --live --coin BTC --testnet

# Backtest on synthetic data
python backtester.py

# Backtest on historical CSV
python -c "
from backtester import BacktestEngine, TickLoader
ticks = TickLoader.from_csv('data/btc_ticks.csv')
# ... attach strategies and run
"
```

Each strategy module is independently runnable:

```bash
python spoofing_counter.py
python funding_arbitrage.py
python adverse_selection.py
```

---

## Architecture

```
[Market Data]
    │
    ├── HyperliquidFeed (live) or TickLoader (backtest)
    │
    ▼
[CentralRunner / BacktestEngine]
    │
    ├── FlowToxicityClassifier (VPIN + Kyle's λ + Rolling ADV)
    │
    ├── Strategy modules (per-tick signals + execution orders)
    │       └── hot_paths (OFI, vol, fill prob — Cython if available)
    │
    ├── CentralRiskManager (pre-flight: halt / shave / fat-finger / cooldown)
    │
    └── AdverseSelectionMonitor (markout PnL, Roll spread, Amihud)
```

---

## Risk manager

`CentralRiskManager` sits between strategy signal generation and order dispatch.

```python
from central_risk_manager import CentralRiskManager, RiskConfig

rm = CentralRiskManager(RiskConfig(
    max_net_position   = 5.0,      # BTC
    max_drawdown_limit = 500.0,    # USD
    daily_loss_limit   = 1000.0,   # USD
    size_decimals      = 4,        # Hyperliquid BTC lot step = 0.0001
    toxicity_cooldown_s = 60.0,
))
```

Per-tick call order:

```python
rm.update_market_state(book.mid)
alive = rm.update_global_pnl(realized, unrealized)   # False = circuit breaker
orders = rm.pre_flight_check(strategy_name, proposed_orders, inventory, vol, regime)
# ... after detecting an adverse fill:
rm.report_toxic_fill(strategy_name)
```

---

## PnL decomposition

All strategies track: spread capture, inventory PnL, adverse selection cost, and funding/basis PnL.
`AdverseSelectionMonitor` adds post-fill markout analysis at 5s / 15s / 30s / 60s / 300s horizons.

---

## Notes

- Liquidation front-running uses only publicly observable order book data and exchange OI metrics.
- Spoofing detection is a counter-strategy tool, not a spoofing implementation.
- Calibrate all rolling windows, thresholds, and `size_decimals` to your venue before going live.
- The Cython extension is optional. All hot paths have a pure Python fallback that runs automatically.
