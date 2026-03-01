"""
MT5 Backtest Environment
==========================
Gymnasium environment that uses the MT5 Strategy Tester via the file bridge.
The EA handles all order execution with real spreads/slippage.
"""

import os
import sys
import logging
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_model.mt5_bridge import MT5Bridge
from rl_model.features import NUM_FEATURES, NUM_MARKET_FEATURES

logger = logging.getLogger(__name__)


class MT5BacktestEnv(gym.Env):
    """
    Gymnasium environment backed by MT5 Strategy Tester.

    The EA (RLBridge.mq5) sends market state → Python computes action →
    EA executes order → Python receives result with real P/L.

    Observation: 14-dim vector (same as simulated TradingEnv)
    Actions:     Discrete(4) — 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
    """

    metadata = {"render_modes": ["human"]}

    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3

    def __init__(
        self,
        bridge_folder="rl_bridge",
        initial_balance=10000.0,
        render_mode=None,
        timeout_sec=10.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.initial_balance = initial_balance

        # Create bridge
        self.bridge = MT5Bridge(
            bridge_folder=bridge_folder,
            timeout_sec=timeout_sec,
        )

        # Spaces (same as simulated env)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(NUM_FEATURES,),
            dtype=np.float32,
        )

        # State tracking
        self._obs_buffer = np.zeros(NUM_FEATURES, dtype=np.float32)
        self._current_state = None
        self._prev_close = None
        self._prev_balance = initial_balance
        self._step_count = 0

        # Episode stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.peak_balance = initial_balance
        self.trade_log = []

        # MA buffer for computing features from raw OHLCV
        self._close_history = []
        self._ma_periods = [5, 10, 20, 100]

    def reset(self, seed=None, options=None):
        """
        Reset: signal EA to restart, wait for first state.
        """
        super().reset(seed=seed)

        # Reset stats
        self._step_count = 0
        self._prev_close = None
        self._prev_balance = self.initial_balance
        self._close_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.peak_balance = self.initial_balance
        self.trade_log = []

        # Signal EA to reset
        self.bridge.send_reset()

        # Wait for first state from EA
        logger.debug("Waiting for first state from EA...")
        state = self.bridge.wait_for_state()
        if state is None:
            logger.error("Timeout waiting for initial state from EA")
            return self._obs_buffer.copy(), self._get_info()

        self._current_state = state
        self._prev_balance = state["balance"]
        self._update_close_history(state["close"])

        obs = self._state_to_obs(state)
        return obs, self._get_info()

    def step(self, action):
        """
        Send action to EA, wait for result and next state.
        """
        self._step_count += 1

        # Send action and get result
        result = self.bridge.step(int(action))
        if result is None:
            logger.warning(f"Step {self._step_count}: timeout from EA")
            return self._obs_buffer.copy(), 0.0, True, False, self._get_info()

        # Extract result
        realized_pnl = result["realized_pnl"]
        balance = result["balance"]
        done = result["done"]
        position = result["position"]

        # Track trades
        if realized_pnl != 0:
            self.total_trades += 1
            if realized_pnl > 0:
                self.winning_trades += 1
                self.total_profit += realized_pnl
            else:
                self.losing_trades += 1
                self.total_loss += abs(realized_pnl)

            self.trade_log.append({
                "step": self._step_count,
                "pnl": round(realized_pnl, 2),
                "balance": round(balance, 2),
            })

        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance

        # Compute reward
        reward = self._compute_reward(
            action, realized_pnl, balance, position
        )
        self._prev_balance = balance

        if done:
            obs = self._obs_buffer.copy()
            return obs, float(reward), True, False, self._get_info()

        # Wait for next state
        state = self.bridge.wait_for_state()
        if state is None:
            return self._obs_buffer.copy(), float(reward), True, False, self._get_info()

        self._current_state = state
        self._update_close_history(state["close"])
        obs = self._state_to_obs(state)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), False, False, self._get_info()

    def _compute_reward(self, action, realized_pnl, balance, position):
        """Same reward logic as simulated env for consistency."""
        reward = 0.0

        # Realized P/L (log return)
        if realized_pnl != 0 and self._prev_balance > 0 and balance > 0:
            reward = np.log(balance / self._prev_balance) * 100

        # Holding penalty
        if position != 0:
            reward -= 0.0001

        # Trade cost (already included by MT5's spread, but add small signal)
        if action in (self.BUY, self.SELL):
            reward -= 0.0025

        # Drawdown penalty
        if self.peak_balance > 0:
            dd = (self.peak_balance - balance) / self.peak_balance
            if dd > 0.05 and reward < 0:
                reward *= 2.0

        return reward

    def _state_to_obs(self, state):
        """Convert MT5 state dict to observation vector."""
        close = state["close"]

        # Compute MAs from close history
        mas = self._compute_mas()
        ma5 = mas.get(5, close)
        ma10 = mas.get(10, close)
        ma20 = mas.get(20, close)
        ma100 = mas.get(100, close)

        # Price ratios
        self._obs_buffer[0] = close / ma5 - 1.0 if ma5 > 0 else 0.0
        self._obs_buffer[1] = close / ma10 - 1.0 if ma10 > 0 else 0.0
        self._obs_buffer[2] = close / ma20 - 1.0 if ma20 > 0 else 0.0
        self._obs_buffer[3] = close / ma100 - 1.0 if ma100 > 0 else 0.0

        # MA diffs
        self._obs_buffer[4] = (ma5 - ma20) / close if close > 0 else 0.0
        self._obs_buffer[5] = (ma10 - ma100) / close if close > 0 else 0.0

        # Returns (approximate)
        ret = 0.0
        if self._prev_close and self._prev_close > 0:
            ret = (close - self._prev_close) / self._prev_close
        self._obs_buffer[6] = np.clip(ret, -0.1, 0.1)
        self._obs_buffer[7] = np.clip(ret, -0.1, 0.1)
        self._obs_buffer[8] = np.clip(ret, -0.1, 0.1)

        # Volatility proxy
        self._obs_buffer[9] = abs(ret) * 100

        # Time features (from timestamp if available)
        self._obs_buffer[10] = 0.0  # hour_sin
        self._obs_buffer[11] = 0.0  # hour_cos

        # Position state
        self._obs_buffer[NUM_MARKET_FEATURES] = float(state["position"])
        norm_pnl = state["unrealized_pnl"] / max(self.initial_balance, 1.0)
        self._obs_buffer[NUM_MARKET_FEATURES + 1] = np.clip(norm_pnl, -1.0, 1.0)

        self._prev_close = close

        # Clip all
        np.clip(self._obs_buffer, -5.0, 5.0, out=self._obs_buffer)

        return self._obs_buffer.copy()

    def _update_close_history(self, close):
        """Track close prices for MA computation."""
        self._close_history.append(close)
        # Keep only last 100 for MA100
        if len(self._close_history) > 200:
            self._close_history = self._close_history[-200:]

    def _compute_mas(self):
        """Compute MAs from close history."""
        result = {}
        for period in self._ma_periods:
            if len(self._close_history) >= period:
                result[period] = np.mean(self._close_history[-period:])
            else:
                result[period] = np.mean(self._close_history) if self._close_history else 0.0
        return result

    def _get_info(self):
        """Episode statistics."""
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        pf = (self.total_profit / self.total_loss) if self.total_loss > 0 else float("inf")
        balance = self._current_state["balance"] if self._current_state else self.initial_balance
        dd = (self.peak_balance - balance) / self.peak_balance if self.peak_balance > 0 else 0.0

        return {
            "step": self._step_count,
            "balance": round(balance, 2),
            "equity": round(self._current_state.get("equity", balance) if self._current_state else balance, 2),
            "position": self._current_state.get("position", 0) if self._current_state else 0,
            "unrealized_pnl": round(self._current_state.get("unrealized_pnl", 0) if self._current_state else 0, 2),
            "total_trades": self.total_trades,
            "win_rate": round(wr, 2),
            "profit_factor": round(pf, 4),
            "drawdown": round(dd, 4),
            "net_profit": round(balance - self.initial_balance, 2),
        }

    def render(self):
        if self._current_state:
            s = self._current_state
            pos_str = {0: "FLAT", 1: "LONG", -1: "SHORT"}.get(s["position"], "?")
            print(
                f"Step {self._step_count:>6d} | "
                f"Close: {s['close']:>8.2f} | "
                f"Pos: {pos_str:>5s} | "
                f"P/L: {s['unrealized_pnl']:>+8.2f} | "
                f"Balance: {s['balance']:>10.2f} | "
                f"Trades: {self.total_trades}"
            )

    def close(self):
        self.bridge.cleanup()
        super().close()
