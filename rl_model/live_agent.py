"""
Live Agent
===========
Wraps the trained PPO model to match Phase 1's get_signal(state) interface.
This module is the bridge between the RL model and the MT5 execution system.

Usage in main.py:
    from rl_model.live_agent import RLAgent
    agent = RLAgent()
    signal = agent.get_signal(state)  # Returns "BUY", "SELL", "CLOSE", or "HOLD"
"""

import os
import sys
import logging

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.features import build_observation

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class RLAgent:
    """
    Wraps the trained PPO model for live trading integration.

    Provides the same get_signal(state) interface as Phase 1's MA crossover,
    making it a drop-in replacement.
    """

    ACTION_MAP = {
        0: "HOLD",
        1: "BUY",
        2: "SELL",
        3: "CLOSE",
    }

    def __init__(self, model_name="best"):
        """
        Load the trained model.

        Args:
            model_name: "best" or "final"
        """
        model_path = os.path.join(MODEL_DIR, f"ppo_xauusd_{model_name}")
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(
                f"Model not found: {model_path}.zip\n"
                "Train first with: python -m rl_model.train"
            )

        logger.info(f"Loading RL model from: {model_path}")
        self.model = PPO.load(model_path)
        logger.info("RL model loaded successfully")

        # Track state for position features
        self._prev_close = None

    def get_signal(self, state):
        """
        Generate a trading signal from the current market state.

        This is the Phase 1 compatible interface.

        Args:
            state: dict with keys:
                - close: current close price
                - ma5, ma10, ma20, ma100: moving average values
                - has_position: bool
                - position_type: 'BUY', 'SELL', or None
                - position_profit: float

        Returns:
            str: "BUY", "SELL", "CLOSE", or "HOLD"
        """
        obs = self._state_to_observation(state)
        action, _ = self.model.predict(obs, deterministic=True)
        signal = self.ACTION_MAP[int(action)]

        # Validate signal against position state
        signal = self._validate_signal(signal, state)

        return signal

    def _state_to_observation(self, state):
        """Convert Phase 1 state dict to RL observation vector."""
        close = state["close"]
        ma5 = state.get("ma5", close)
        ma10 = state.get("ma10", close)
        ma20 = state.get("ma20", close)
        ma100 = state.get("ma100", close)

        # Handle NaN
        if ma5 != ma5: ma5 = close
        if ma10 != ma10: ma10 = close
        if ma20 != ma20: ma20 = close
        if ma100 != ma100: ma100 = close

        # Price ratios
        close_ma5 = close / ma5 - 1.0 if ma5 > 0 else 0.0
        close_ma10 = close / ma10 - 1.0 if ma10 > 0 else 0.0
        close_ma20 = close / ma20 - 1.0 if ma20 > 0 else 0.0
        close_ma100 = close / ma100 - 1.0 if ma100 > 0 else 0.0

        # MA diffs
        ma5_ma20 = (ma5 - ma20) / close if close > 0 else 0.0
        ma10_ma100 = (ma10 - ma100) / close if close > 0 else 0.0

        # Momentum (approximate with single-step return)
        if self._prev_close and self._prev_close > 0:
            ret = (close - self._prev_close) / self._prev_close
        else:
            ret = 0.0
        self._prev_close = close

        # Position state
        pos_flag = 0.0
        if state.get("position_type") == "BUY":
            pos_flag = 1.0
        elif state.get("position_type") == "SELL":
            pos_flag = -1.0

        pnl_norm = state.get("position_profit", 0.0) / 10000.0  # Normalize
        pnl_norm = np.clip(pnl_norm, -1.0, 1.0)

        # Build 14-dim observation (matching training features)
        market_features = np.array([
            close_ma5, close_ma10, close_ma20, close_ma100,
            ma5_ma20, ma10_ma100,
            ret, ret, ret,  # Same return for all lookbacks (approximation)
            abs(ret) * 100,  # Volatility proxy
            0.0, 0.0,  # Hour features (0 for live — can enhance later)
        ], dtype=np.float32)

        obs = build_observation(market_features, pos_flag, pnl_norm)
        return obs

    def _validate_signal(self, signal, state):
        """Prevent invalid actions (e.g., BUY when already long)."""
        has_pos = state.get("has_position", False)
        pos_type = state.get("position_type")

        if signal == "BUY" and has_pos:
            return "HOLD"
        if signal == "SELL" and has_pos:
            return "HOLD"
        if signal == "CLOSE" and not has_pos:
            return "HOLD"

        return signal


# ── Convenience function ─────────────────────────

_agent_singleton = None


def get_signal(state):
    """
    Module-level get_signal function for direct import in main.py.
    Lazy-loads the model on first call.
    """
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = RLAgent()
    return _agent_singleton.get_signal(state)
