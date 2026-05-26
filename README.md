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
| `runner.py` | Central orchestrator - simulation and live Hyperliquid modes |
| `hyperliquid_feed.py` | Live WebSocket feed with `asyncio.Queue` + `call_soon_threadsafe` |
| `central_risk_manager.py` | Pre-trade risk gate: circuit breaker, sizing shaver, fat-finger, toxicity cooldown |
| `backtester.py` | Tick-by-tick L2 replay with FIFO queue simulation, latency modeling, and PnL attribution |
| `adverse_selection.py` | Markout PnL, effective/realized spread decomposition, Roll estimator, Amihud ratio |
| `rl_param_tuner.py` | SAC/PPO agent that optimizes Avellaneda-Stoikov parameters + toxic flow thresholds in real time |
| `hot_paths.py` | Auto-selecting wrapper - Cython extension or pure Python fallback |
| `_hot_paths.pyx` | Cython hot paths: OFI, VPIN bucket fill, Kyle's λ OLS, Poisson fill prob, lot floor |
| `_hot_paths_pure.py` | Pure Python fallback - identical API to the compiled extension |
| `setup_hot_paths.py` | Build script for the Cython extension |

---

## Setup

```bash
pip install -r requirements.txt

# Optional: compile Cython hot paths (~10-100x speedup on inner loops)
python setup_hot_paths.py build_ext --inplace

# Optional: RL parameter tuner
pip install gymnasium stable-baselines3
```

---

## Run

```bash
# Simulation
python runner.py

# Live
python runner.py --live --coin BTC

# Testnet
python runner.py --live --coin BTC --testnet

# Backtest - simple mode
python backtester.py

# Backtest - pro mode (FIFO queue + latency)
python -c "
from backtester import ProBacktestEngine, TickLoader, LatencyConfig, ProbQueueCancelModel
ticks  = TickLoader.from_csv('data/btc_ticks.csv')
engine = ProBacktestEngine(
    strategies={...},
    latency_config=LatencyConfig(feed_latency_us=150, order_latency_us=600),
    cancel_model=ProbQueueCancelModel(),
)
engine.run(ticks).print_summary()
"

# RL parameter tuner - pipeline demo (no gym required)
python rl_param_tuner.py

# RL parameter tuner - train SAC agent
python -c "
from rl_param_tuner import train_agent, TrainConfig
agent = train_agent(TrainConfig(algorithm='SAC', total_timesteps=500_000))
agent.save('models/as_rl_agent')
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
[CentralRunner / BacktestEngine / ProBacktestEngine]
    │
    ├── FlowToxicityClassifier (VPIN + Kyle's λ + Rolling ADV)
    │
    ├── Strategy modules (per-tick signals + execution orders)
    │       ├── hot_paths (OFI, vol, fill prob - Cython if available)
    │       └── RLAugmentedGuerrillaStrategy ← ASParamAgent (SAC/PPO)
    │               └── optimizes γ, spread, toxicity threshold, size
    │
    ├── CentralRiskManager (pre-flight: halt / shave / fat-finger / cooldown)
    │
    ├── FIFOQueueSimulator + LatencySimulator  [ProBacktestEngine only]
    │       ├── ReduceRatioCancelModel / ProbQueueCancelModel
    │       ├── Iceberg detection via level replenishment
    │       └── Log-normal order flight time (feed + order latency)
    │
    └── AdverseSelectionMonitor (markout PnL, Roll spread, Amihud)
```

---

## Backtester modes

| Feature | `BacktestEngine` | `ProBacktestEngine` |
|---------|-----------------|---------------------|
| Tick-by-tick L2 replay | ✓ | ✓ |
| Passive fill simulation | conservative queue share | FIFO queue position |
| Cancellation model | - | ReduceRatio / ProbQueue |
| Iceberg detection | - | replenishment pattern |
| Feed latency (stale book) | - | log-normal, configurable |
| Order flight time | - | log-normal, configurable |
| Latency displacement tracking | - | ✓ |
| Queue advancement metrics | - | ✓ |

> **Note on L2 vs L3.** Hyperliquid's public feed is L2. `ProBacktestEngine` extracts maximum fidelity from L2: FIFO position is estimated probabilistically from depth and inferred cancellations. True event-by-event L3 replay would require exchange-side data not publicly available.

---

## RL parameter tuner

`rl_param_tuner.py` trains a SAC agent to continuously optimize the Avellaneda-Stoikov parameters and toxic flow thresholds of `AdaptiveGuerrillaStrategy`. The agent does not replace the strategy - it tunes it.

**State (14 features):** AS model state (γ, σ, T-t), toxicity metrics (VPIN, Kyle's λ, composite score), inventory skew, cancel rate, spread bps, book imbalance, realized PnL, adverse selection cost, time to funding.

**Actions (4 continuous):** `gamma_multiplier` ∈ [0.3, 3.0], `spread_multiplier` ∈ [0.5, 2.0], `toxicity_threshold` ∈ [0.25, 0.85], `size_multiplier` ∈ [0.3, 1.5].

**Reward:** ΔPnL − inventory risk penalty − adverse selection cost − drawdown penalty − toxic hold penalty.

```python
from rl_param_tuner import train_agent, compare_baseline, TrainConfig

agent = train_agent(TrainConfig(algorithm="SAC", total_timesteps=500_000))
compare_baseline(agent)   # prints RL-tuned vs fixed-param PnL side by side
```

The pipeline (observation → action → params) works without gymnasium. Only training requires `pip install gymnasium stable-baselines3`.

---

## Risk manager

```python
from central_risk_manager import CentralRiskManager, RiskConfig

rm = CentralRiskManager(RiskConfig(
    max_net_position    = 5.0,     # BTC
    max_drawdown_limit  = 500.0,   # USD
    daily_loss_limit    = 1000.0,  # USD
    size_decimals       = 4,       # Hyperliquid BTC lot step = 0.0001
    toxicity_cooldown_s = 60.0,
))
```

Per-tick call order:

```python
rm.update_market_state(book.mid)
alive  = rm.update_global_pnl(realized, unrealized)  # False = circuit breaker
orders = rm.pre_flight_check(strategy_name, proposed_orders, inventory, vol, regime)
rm.report_toxic_fill(strategy_name)                  # after detecting adverse fill
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
- The Cython extension is optional. All hot paths fall back to pure Python automatically.
- `cancel_ratio` in `ReduceRatioCancelModel` should be calibrated per venue and distance-to-BBO. Typical range for crypto perps: 0.10-0.35.
- SAC is recommended over PPO for this problem - off-policy learning handles the non-stationarity of market regimes better.

*Rest in peace Toto. This is in your honor.*
