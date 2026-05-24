# game-theory-trading-strats

Game-theoretic microstructure strategies for crypto perpetuals.
Tested on Hyperliquid; adaptable to Binance Perps, dYdX, and similar venues.

---

## Modules

| Module | Game | Edge |
|--------|------|------|
| `spoofing_counter.py` | Crawford-Sobel Signaling | Detect fake depth, fade the illusion |
| `predatory_liquidity.py` | Stackelberg Coordination | Join or fade stop cascades |
| `info_asymmetry.py` | Glosten-Milgrom Adverse Selection | VPIN, Kyle's λ, flow toxicity |
| `queue_warfare.py` | FIFO Queue Leadership | Iceberg detection, OFI-triggered cancel |
| `funding_arbitrage.py` | Convergence Timing Game | Pre-print funding capture |
| `liquidation_frontrun.py` | Dominated Strategy Exploitation | Front-run deterministic liq engines |
| `adaptive_guerrilla.py` | Avellaneda-Stoikov + toxic flow cancel | Inventory skew, adverse selection deflection |

---

## Run

```bash
# Simulation
python runner.py

# Live (real money)
python runner.py --live --coin BTC

# Testnet
python runner.py --live --coin BTC --testnet
```

Each module is independently runnable:

```bash
python spoofing_counter.py
python funding_arbitrage.py
```

---

## PnL decomposition

All strategies track: spread capture, inventory PnL, adverse selection cost, and funding/basis PnL.

---

## Notes

- Liquidation front-running uses only publicly observable OB data and exchange OI metrics.
- Spoofing detection is a counter-strategy tool, not a spoofing implementation.
- Calibrate all rolling windows and thresholds to your venue before going live.
- Liquidation front-running uses only publicly observable OB data and exchange OI metrics.
- Spoofing detection is a counter-strategy tool, not a spoofing implementation.
- Calibrate all rolling windows and thresholds to your venue before going live.
