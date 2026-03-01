"""
Trading Environment (Optimized)
================================
Custom Gymnasium environment for XAUUSD M1 trading.
Optimized: pre-allocated arrays, minimal per-step allocations, 
simplified reward for speed.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_model.features import precompute_features, NUM_FEATURES, NUM_MARKET_FEATURES


class TradingEnv(gym.Env):
    """
    XAUUSD M1 Trading Environment — Optimized for fast training.

    Observation: 14-dimensional vector (market features + position state)
    Actions:     Discrete(4) — 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
    Episode:     One full pass through the data.
    """

    metadata = {"render_modes": ["human"]}

    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3

    def __init__(
        self,
        df,
        initial_balance=10000.0,
        lot_size=0.01,
        contract_size=100,
        spread_points=25,
        max_position=1,
        render_mode=None,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Pre-compute and store as contiguous numpy arrays
        self.close_prices = np.ascontiguousarray(df["Close"].values, dtype=np.float64)
        self.n_steps = len(df)
        self.market_features = np.ascontiguousarray(precompute_features(df))

        # Trading params
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.contract_size = contract_size
        self.spread = spread_points * 0.01
        self.lot_contract = lot_size * contract_size  # Pre-compute

        # Spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(NUM_FEATURES,),
            dtype=np.float32,
        )

        # Pre-allocate observation buffer
        self._obs_buffer = np.zeros(NUM_FEATURES, dtype=np.float32)

        self._reset_state()

    def _reset_state(self):
        """Initialize/reset all episode state variables."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0       # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.peak_balance = self.initial_balance
        self.trade_log = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs_fast(), self._get_info()

    def step(self, action):
        price = self.close_prices[self.current_step]
        realized_pnl = 0.0
        reward = 0.0
        trade_opened = False

        # ── Execute action ─────────────────────
        if action == self.BUY and self.position == 0:
            self.position = 1
            self.entry_price = price + self.spread
            trade_opened = True

        elif action == self.SELL and self.position == 0:
            self.position = -1
            self.entry_price = price
            trade_opened = True

        elif action == self.CLOSE and self.position != 0:
            realized_pnl = self._close_position(price)

        # ── Update unrealized P/L ──────────────
        if self.position == 1:
            self.unrealized_pnl = (price - self.entry_price) * self.lot_contract
        elif self.position == -1:
            self.unrealized_pnl = (self.entry_price - price - self.spread) * self.lot_contract
        else:
            self.unrealized_pnl = 0.0

        # ── Compute reward (simplified for speed) ──
        if realized_pnl != 0:
            # Log return on trade close (risk-sensitive)
            old_bal = self.balance - realized_pnl
            if old_bal > 0 and self.balance > 0:
                reward = np.log(self.balance / old_bal) * 100
            else:
                reward = np.sign(realized_pnl) * 0.1

        if self.position != 0:
            reward -= 0.0001  # Holding penalty

        if trade_opened:
            reward -= self.spread * self.lot_contract * 0.01  # Spread cost

        # Drawdown penalty
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        if self.peak_balance > 0:
            dd = (self.peak_balance - self.balance) / self.peak_balance
            if dd > 0.05 and reward < 0:
                reward *= 2.0

        # ── Advance ────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1

        # Force close at end
        if terminated and self.position != 0:
            ep = self.close_prices[min(self.current_step, self.n_steps - 1)]
            final_pnl = self._close_position(ep)
            if final_pnl != 0:
                old_b = self.balance - final_pnl
                if old_b > 0 and self.balance > 0:
                    reward += np.log(self.balance / old_b) * 100

        # Bankruptcy
        if self.balance <= 0:
            terminated = True
            reward -= 10.0

        obs = self._get_obs_fast()
        return obs, float(reward), terminated, False, self._get_info()

    def _close_position(self, current_price):
        """Close position, return realized P/L."""
        if self.position == 1:
            pnl = (current_price - self.entry_price) * self.lot_contract
        elif self.position == -1:
            pnl = (self.entry_price - current_price - self.spread) * self.lot_contract
        else:
            return 0.0

        self.balance += pnl
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        elif pnl < 0:
            self.losing_trades += 1
            self.total_loss += abs(pnl)

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        self.trade_log.append({
            "step": self.current_step,
            "type": "LONG" if self.position == 1 else "SHORT",
            "entry": self.entry_price,
            "exit": current_price,
            "pnl": round(pnl, 2),
            "balance": round(self.balance, 2),
        })

        self.position = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        return pnl

    def _get_obs_fast(self):
        """Build observation — minimal allocation using pre-allocated buffer."""
        step = min(self.current_step, self.n_steps - 1)
        self._obs_buffer[:NUM_MARKET_FEATURES] = self.market_features[step]
        self._obs_buffer[NUM_MARKET_FEATURES] = float(self.position)
        norm_pnl = self.unrealized_pnl / self.initial_balance
        self._obs_buffer[NUM_MARKET_FEATURES + 1] = max(-1.0, min(1.0, norm_pnl))
        return self._obs_buffer.copy()  # SB3 needs a copy

    def _get_info(self):
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        pf = (self.total_profit / self.total_loss) if self.total_loss > 0 else float("inf")
        dd = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0.0
        return {
            "step": self.current_step,
            "balance": round(self.balance, 2),
            "equity": round(self.balance + self.unrealized_pnl, 2),
            "position": self.position,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "total_trades": self.total_trades,
            "win_rate": round(wr, 2),
            "profit_factor": round(pf, 4),
            "drawdown": round(dd, 4),
            "net_profit": round(self.balance - self.initial_balance, 2),
        }

    def render(self):
        info = self._get_info()
        pos_str = {0: "FLAT", 1: "LONG", -1: "SHORT"}[self.position]
        price = self.close_prices[min(self.current_step, self.n_steps - 1)]
        print(
            f"Step {self.current_step:>6d} | Price: {price:>8.2f} | "
            f"Pos: {pos_str:>5s} | P/L: {self.unrealized_pnl:>+8.2f} | "
            f"Balance: {self.balance:>10.2f} | Trades: {self.total_trades}"
        )


def make_env(df, **kwargs):
    return TradingEnv(df, **kwargs)
