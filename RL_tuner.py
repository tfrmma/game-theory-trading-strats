from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from init import OrderBook, Trade, Side, InventoryState
from adaptive_guerrilla import AdaptiveGuerrillaStrategy
from info_asymmetry import FlowToxicityClassifier
from backtester import TickLoader, HistoricalTick

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces
    _GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces as gym_spaces
        _GYM = True
    except ImportError:
        _GYM = False
        gym = None

try:
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    _SB3 = True
except ImportError:
    _SB3 = False

OBS_DIM = 14
ACT_DIM = 4

# RL outputs in [-1, 1]; mapped to these real ranges
ACT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "gamma_multiplier":   (0.3, 3.0),
    "spread_multiplier":  (0.5, 2.0),
    "toxicity_threshold": (0.25, 0.85),
    "size_multiplier":    (0.3, 1.5),
}

@dataclass
class RLParams:
    gamma_multiplier:   float = 1.0   # scales AS risk aversion
    spread_multiplier:  float = 1.0   # scales optimal spread
    toxicity_threshold: float = 0.55  # cancel threshold for toxic flow
    size_multiplier:    float = 1.0   # scales base order size

@dataclass
class RewardConfig:
    inv_penalty_coef:   float = 0.10  # (inventory/max_inv)^2 penalty
    adv_sel_coef:       float = 0.50  # adverse selection cost delta penalty
    dd_coef:            float = 2.00  # drawdown penalty multiplier
    dd_free_threshold:  float = 50.0  # drawdown below this is unpunished
    toxic_hold_penalty: float = 3.00  # penalty for holding inventory in TOXIC regime

@dataclass
class TrainConfig:
    algorithm:        str   = "SAC"       # "SAC" | "PPO"
    total_timesteps:  int   = 500_000
    ticks_per_step:   int   = 20          # market ticks per RL decision
    episode_ticks:    int   = 2_000       # ticks per episode
    eval_episodes:    int   = 5
    eval_freq:        int   = 10_000
    seed:             int   = 42
    tensorboard_log:  Optional[str] = None

class RLAugmentedGuerrillaStrategy(AdaptiveGuerrillaStrategy):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._base_gamma      = self.as_model.gamma
        self._base_size_usd   = self.base_size_usd
        self._rl_params       = RLParams()

    def apply_rl_params(self, params: RLParams) -> None:
        self.as_model.gamma            = self._base_gamma * params.gamma_multiplier
        self.toxicity_cancel_threshold = params.toxicity_threshold
        self.base_size_usd             = self._base_size_usd * params.size_multiplier
        self._rl_params                = params

    def reset(self) -> None:
        self.inventory         = InventoryState()
        self._active_quotes    = {}
        self._cancel_log       = []
        self._mid_history.clear()
        self.as_model.reset_epoch()
        self.apply_rl_params(RLParams())

def build_observation(
    strategy:       RLAugmentedGuerrillaStrategy,
    toxicity_state: dict,
    book:           OrderBook,
    max_inventory:  float,
    pnl_norm_scale: float = 100.0,
    time_to_funding_s: float = 0.5,
) -> np.ndarray:
    inv    = strategy.inventory
    as_m   = strategy.as_model

    # AS model state
    gamma_norm     = np.clip(as_m.gamma / 0.5, 0, 4)
    sigma_norm     = np.clip(as_m.sigma / 0.002, 0, 4)
    t_remaining    = np.clip(as_m.remaining_t / as_m.T, 0, 1)

    # Toxicity
    vpin           = float(toxicity_state.get("vpin") or 0)
    kyle_lam       = np.clip(float(toxicity_state.get("lambda", 0)) / 0.05, 0, 4)
    tox_score      = float(toxicity_state.get("toxicity_score", 0))

    # Inventory
    inv_skew       = np.clip(inv.net_position / max(max_inventory, 1e-9), -1, 1)
    cancel_rate    = np.clip(strategy.cancel_rate, 0, 1)

    # Market microstructure
    spread_bps     = np.clip(book.spread_bps / 10.0, 0, 4)
    imbalance      = np.clip(book.imbalance(5), -1, 1)
    vol_norm       = sigma_norm  # reuse realized vol from AS model

    # PnL / cost
    pnl_norm       = np.clip(inv.realized_pnl / pnl_norm_scale, -2, 2)
    adv_sel_norm   = np.clip(inv.adverse_selection_cost / pnl_norm_scale, 0, 4)

    # Funding horizon
    funding_norm   = np.clip(1.0 - time_to_funding_s, 0, 1)

    obs = np.array([
        gamma_norm, sigma_norm, t_remaining,
        vpin, kyle_lam, tox_score,
        inv_skew, cancel_rate,
        spread_bps, imbalance, vol_norm,
        pnl_norm, adv_sel_norm,
        funding_norm,
    ], dtype=np.float32)

    assert len(obs) == OBS_DIM
    return obs

def map_action_to_params(action: np.ndarray) -> RLParams:
    def scale(val: float, low: float, high: float) -> float:
        return float(low + (np.clip(val, -1, 1) + 1) / 2 * (high - low))

    keys = list(ACT_BOUNDS.keys())
    vals = {k: scale(action[i], *ACT_BOUNDS[k]) for i, k in enumerate(keys)}
    return RLParams(**vals)

def _require_gym() -> None:
    if not _GYM:
        raise ImportError(
            "gymnasium (or gym) is required: pip install gymnasium stable-baselines3"
        )

if _GYM:
    class ASParamEnv(gym.Env):

        metadata = {"render_modes": []}

        def __init__(
            self,
            ticks_per_step:   int   = 20,
            episode_ticks:    int   = 2_000,
            max_inventory:    float = 5.0,
            reward_config:    Optional[RewardConfig] = None,
            ticks:            Optional[List[HistoricalTick]] = None,
            seed:             int   = 42,
        ) -> None:
            super().__init__()
            self._ticks_per_step = ticks_per_step
            self._episode_ticks  = episode_ticks
            self._max_inv        = max_inventory
            self._rew_cfg        = reward_config or RewardConfig()
            self._source_ticks   = ticks
            self._rng            = np.random.default_rng(seed)

            self.observation_space = gym_spaces.Box(
                low=-4.0, high=4.0, shape=(OBS_DIM,), dtype=np.float32
            )
            self.action_space = gym_spaces.Box(
                low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
            )

            self._strategy: Optional[RLAugmentedGuerrillaStrategy] = None
            self._toxicity: Optional[FlowToxicityClassifier] = None
            self._ticks:    List[HistoricalTick] = []
            self._tick_idx: int = 0
            self._hwm:      float = 0.0
            self._last_tox: dict = {}

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)

            self._strategy = RLAugmentedGuerrillaStrategy(
                max_inventory=self._max_inv,
                base_size_usd=8_000.0,
            )
            self._toxicity = FlowToxicityClassifier()
            self._tick_idx = 0
            self._hwm      = 0.0
            self._last_tox = {}

            self._ticks = self._load_ticks()
            obs = self._observe()
            return obs, {}

        def step(self, action: np.ndarray):
            params = map_action_to_params(action)
            self._strategy.apply_rl_params(params)

            pnl_before     = self._strategy.inventory.realized_pnl
            adv_sel_before = self._strategy.inventory.adverse_selection_cost

            end_idx = min(self._tick_idx + self._ticks_per_step, len(self._ticks))
            for tick in self._ticks[self._tick_idx:end_idx]:
                self._run_tick(tick)
            self._tick_idx = end_idx

            reward     = self._compute_reward(pnl_before, adv_sel_before)
            terminated = self._tick_idx >= len(self._ticks)
            obs        = self._observe()

            info = {
                "pnl":         self._strategy.inventory.realized_pnl,
                "inventory":   self._strategy.inventory.net_position,
                "cancel_rate": self._strategy.cancel_rate,
                "toxicity":    self._last_tox.get("toxicity_score", 0),
                "params":      params,
            }
            return obs, reward, terminated, False, info

        def _run_tick(self, tick: HistoricalTick) -> None:
            self._last_tox = self._toxicity.update(tick.book, tick.trades)
            self._strategy.update(tick.book, tick.trades)
            pnl = self._strategy.inventory.realized_pnl
            self._hwm = max(self._hwm, pnl)

        def _observe(self) -> np.ndarray:
            if not self._ticks or self._tick_idx >= len(self._ticks):
                return np.zeros(OBS_DIM, dtype=np.float32)
            tick = self._ticks[min(self._tick_idx, len(self._ticks) - 1)]
            return build_observation(
                strategy=self._strategy,
                toxicity_state=self._last_tox,
                book=tick.book,
                max_inventory=self._max_inv,
            )

        def _compute_reward(self, pnl_before: float, adv_sel_before: float) -> float:
            inv        = self._strategy.inventory
            pnl_delta  = inv.realized_pnl - pnl_before
            adv_delta  = inv.adverse_selection_cost - adv_sel_before
            inv_skew   = abs(inv.net_position) / max(self._max_inv, 1e-9)
            drawdown   = self._hwm - inv.realized_pnl
            tox        = float(self._last_tox.get("toxicity_score", 0))

            reward  = pnl_delta
            reward -= self._rew_cfg.inv_penalty_coef * inv_skew ** 2
            reward -= self._rew_cfg.adv_sel_coef * adv_delta

            if drawdown > self._rew_cfg.dd_free_threshold:
                reward -= self._rew_cfg.dd_coef * (drawdown - self._rew_cfg.dd_free_threshold)

            if abs(inv.net_position) > 0.1 and tox > 0.65:
                reward -= self._rew_cfg.toxic_hold_penalty * tox

            return float(reward)

        def _load_ticks(self) -> List[HistoricalTick]:
            if self._source_ticks:
                start = int(self._rng.integers(0, max(1, len(self._source_ticks) - self._episode_ticks)))
                return self._source_ticks[start: start + self._episode_ticks]
                informed_ramp = bool(self._rng.random() > 0.5)
            seed = int(self._rng.integers(0, 100_000))
            return TickLoader.synthetic(
                n_ticks=self._episode_ticks,
                informed_ramp=informed_ramp,
                seed=seed,
            )

else:
    # Stub so the module imports cleanly without gym
    class ASParamEnv:  # type: ignore
        def __init__(self, *args, **kwargs):
            _require_gym()

class ASParamAgent:

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self._model = None

    def train(
        self,
        env:      "ASParamEnv",
        eval_env: Optional["ASParamEnv"] = None,
    ) -> None:
        if not _SB3:
            raise ImportError("stable-baselines3 required: pip install stable-baselines3")

        algo_cls = SAC if self.config.algorithm == "SAC" else PPO
        wrapped  = Monitor(env)

        callbacks = []
        if eval_env is not None:
            stop_cb = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
            eval_cb = EvalCallback(
                Monitor(eval_env),
                callback_on_new_best=stop_cb,
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.eval_episodes,
                verbose=1,
            )
            callbacks.append(eval_cb)

        self._model = algo_cls(
            policy="MlpPolicy",
            env=wrapped,
            seed=self.config.seed,
            tensorboard_log=self.config.tensorboard_log,
            verbose=1,
        )
        logger.info("Training %s for %d timesteps", self.config.algorithm, self.config.total_timesteps)
        self._model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks or None,
        )

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> RLParams:
        if self._model is None:
            return RLParams()
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return map_action_to_params(action)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("No model to save — call train() first.")
        self._model.save(path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        if not _SB3:
            raise ImportError("stable-baselines3 required: pip install stable-baselines3")
        algo_cls   = SAC if self.config.algorithm == "SAC" else PPO
        self._model = algo_cls.load(path)
        logger.info("Model loaded from %s", path)

def train_agent(
    config:       TrainConfig = TrainConfig(),
    ticks:        Optional[List[HistoricalTick]] = None,
    reward_config: Optional[RewardConfig] = None,
) -> ASParamAgent:
    _require_gym()
    if not _SB3:
        raise ImportError("stable-baselines3 required: pip install stable-baselines3")

    env = ASParamEnv(
        ticks_per_step=config.ticks_per_step,
        episode_ticks=config.episode_ticks,
        reward_config=reward_config,
        ticks=ticks,
        seed=config.seed,
    )
    eval_env = ASParamEnv(
        ticks_per_step=config.ticks_per_step,
        episode_ticks=config.episode_ticks,
        reward_config=reward_config,
        ticks=ticks,
        seed=config.seed + 1,
    )
    agent = ASParamAgent(config)
    agent.train(env, eval_env)
    return agent

def compare_baseline(
    agent:    ASParamAgent,
    n_ticks:  int = 2_000,
    seed:     int = 99,
) -> Dict[str, float]:
    ticks = TickLoader.synthetic(n_ticks=n_ticks, seed=seed)

    def run(use_agent: bool) -> Dict[str, float]:
        strat = RLAugmentedGuerrillaStrategy(max_inventory=5.0)
        tox   = FlowToxicityClassifier()
        for tick in ticks:
            tox_state = tox.update(tick.book, tick.trades)
            if use_agent:
                obs    = build_observation(strat, tox_state, tick.book, max_inventory=5.0)
                params = agent.predict(obs)
                strat.apply_rl_params(params)
            strat.update(tick.book, tick.trades)
        inv = strat.inventory
        return {
            "realized_pnl":        inv.realized_pnl,
            "adverse_sel_cost":    inv.adverse_selection_cost,
            "cancel_rate":         strat.cancel_rate,
        }

    baseline = run(use_agent=False)
    tuned    = run(use_agent=True)

    print("\n" + "="*55)
    print(f"{'Metric':<25} {'Baseline':>12} {'RL-Tuned':>12}")
    print("-"*55)
    for k in baseline:
        print(f"{k:<25} {baseline[k]:>12.4f} {tuned[k]:>12.4f}")
    print("="*55 + "\n")

    return {"baseline": baseline, "tuned": tuned}

def demo_pipeline(n_ticks: int = 200) -> None:
    rng    = np.random.default_rng(42)
    ticks  = TickLoader.synthetic(n_ticks=n_ticks, seed=42)
    strat  = RLAugmentedGuerrillaStrategy(max_inventory=5.0)
    tox    = FlowToxicityClassifier()

    print("\n" + "="*60)
    print("RL PARAM TUNER — PIPELINE DEMO")
    print("="*60)

    for i, tick in enumerate(ticks):
        tox_state = tox.update(tick.book, tick.trades)
        obs       = build_observation(strat, tox_state, tick.book, max_inventory=5.0)

        # Simulate a random agent action (replace with agent.predict(obs) in training)
        raw_action = rng.uniform(-1, 1, ACT_DIM).astype(np.float32)
        params     = map_action_to_params(raw_action)
        strat.apply_rl_params(params)
        strat.update(tick.book, tick.trades)

        if i % 50 == 0:
            inv = strat.inventory
            print(
                f"[{i:3d}] γ={strat.as_model.gamma:.3f} "
                f"tox_thr={strat.toxicity_cancel_threshold:.2f} "
                f"size_usd={strat.base_size_usd:.0f} | "
                f"inv={inv.net_position:.3f} "
                f"pnl={inv.realized_pnl:.4f} "
                f"tox={tox_state.get('toxicity_score', 0):.2f}"
            )

    print(f"\nobs shape={obs.shape} | obs[:4]={obs[:4].round(3)}")
    print(f"action range: gamma_mult=[{ACT_BOUNDS['gamma_multiplier']}] "
          f"tox_thr=[{ACT_BOUNDS['toxicity_threshold']}]")
    print(f"\ngym available:  {_GYM}")
    print(f"sb3 available:  {_SB3}")
    if not _GYM:
        print("\nTo train: pip install gymnasium stable-baselines3")
        print("Then:     from rl_param_tuner import train_agent, TrainConfig")
        print("          agent = train_agent(TrainConfig(algorithm='SAC'))")
    print("="*60 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    demo_pipeline()
