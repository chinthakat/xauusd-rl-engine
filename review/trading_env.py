"""
Trading Environment (v3 — Post Review)
========================================
Custom Gymnasium environment for XAUUSD M1 trading.

Review fixes applied:
  1. Simplified reward: only log-return, holding cost, spread, drawdown
  2. Removed: inactivity penalty, churning penalty, smart hold, trade bonus
  3. Added randomization: episode start, spread, slippage
  4. Equity curve tracking every step
  5. Proper bid/ask with dynamic spread
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_model.features import precompute_features, NUM_FEATURES, NUM_MARKET_FEATURES


class TradingEnv(gym.Env):
    """
    XAUUSD M1 Trading Environment.

    Observation: 14-dimensional vector (market features + position state)
    Actions:     Discrete(4) — 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE

    Position rules:
      - Only 1 position at a time
      - BUY while SHORT → close short, open long (flip)
      - SELL while LONG → close long, open short (flip)
      - Minimum hold time before close/flip
      - Margin required = price × lots × contract_size / leverage
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
        leverage=100,
        max_position=1,
        render_mode=None,
        # Randomization (anti-overfitting)
        random_start=True,          # Start at random bar each episode
        random_spread=True,         # Randomize spread ±30%
        random_slippage=True,       # Add random slippage 0-3 points
        episode_length=None,        # Fixed episode length (None = full)
    ):
        super().__init__()
        self.render_mode = render_mode

        # Pre-compute and store as contiguous numpy arrays
        self.close_prices = np.ascontiguousarray(df["Close"].values, dtype=np.float64)
        self.n_bars = len(df)
        self.market_features = np.ascontiguousarray(precompute_features(df))

        # Trading params
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.contract_size = contract_size
        self.leverage = leverage
        self.base_spread = spread_points * 0.01
        self.lot_contract = lot_size * contract_size

        # Anti-churning
        self.min_hold_steps = 30

        # Randomization settings
        self.random_start = random_start
        self.random_spread = random_spread
        self.random_slippage = random_slippage
        self.episode_length = episode_length

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
        # Random start index (anti-overfitting)
        if self.random_start and self.n_bars > 1000:
            max_start = self.n_bars - 500  # Leave at least 500 bars
            if self.episode_length and self.episode_length < self.n_bars:
                max_start = self.n_bars - self.episode_length
            self.start_idx = self.np_random.integers(0, max(1, max_start))
        else:
            self.start_idx = 0

        # Episode length
        if self.episode_length:
            self.n_steps = min(self.episode_length, self.n_bars - self.start_idx)
        else:
            self.n_steps = self.n_bars - self.start_idx

        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin_used = 0.0
        self.free_margin = self.initial_balance

        # Position
        self.position = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_position = 0

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.peak_balance = self.initial_balance
        self.peak_equity = self.initial_balance
        self.trade_log = []

        # Equity curve (tracked every step for proper metrics)
        self.equity_curve = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs_fast(), self._get_info()

    # ── Helpers ────────────────────────────────────

    def _get_spread(self):
        """Get current spread, optionally randomized."""
        if self.random_spread:
            # Randomize ±30% around base
            return self.base_spread * (0.7 + self.np_random.random() * 0.6)
        return self.base_spread

    def _get_slippage(self):
        """Get random slippage in price units."""
        if self.random_slippage:
            return self.np_random.random() * 0.03  # 0-3 points
        return 0.0

    def _can_open(self, price):
        """Check if we have enough free margin."""
        required = (price * self.lot_size * self.contract_size) / self.leverage
        return self.free_margin >= required

    def _can_close(self):
        """Check if position has been held long enough."""
        return self.steps_in_position >= self.min_hold_steps

    def _open_position(self, direction, price):
        """Open a new position."""
        slippage = self._get_slippage()
        if direction == 1:
            price += slippage  # Worse fill for buy
        else:
            price -= slippage  # Worse fill for sell

        self.position = direction
        self.entry_price = price
        self.margin_used = (price * self.lot_size * self.contract_size) / self.leverage
        self.free_margin = self.equity - self.margin_used
        self.steps_in_position = 0

    def _close_position(self, current_price):
        """Close position, return realized P/L."""
        slippage = self._get_slippage()
        if self.position == 1:
            current_price -= slippage  # Worse fill for closing long
            pnl = (current_price - self.entry_price) * self.lot_contract
        elif self.position == -1:
            current_price += slippage  # Worse fill for closing short
            pnl = (self.entry_price - current_price) * self.lot_contract
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
            "entry": round(self.entry_price, 2),
            "exit": round(current_price, 2),
            "pnl": round(pnl, 2),
            "balance": round(self.balance, 2),
        })

        self.position = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.margin_used = 0.0
        self.free_margin = self.balance
        return pnl

    # ── Core Step ──────────────────────────────────

    def step(self, action):
        bar_idx = self.start_idx + self.current_step
        price = self.close_prices[bar_idx]
        spread = self._get_spread()
        ask = price + spread
        bid = price

        realized_pnl = 0.0
        reward = 0.0
        trade_opened = False
        trade_closed = False

        # ── Execute action ─────────────────────────

        if action == self.BUY:
            if self.position == 0:
                if self._can_open(ask):
                    self._open_position(1, ask)
                    trade_opened = True
            elif self.position == -1 and self._can_close():
                realized_pnl = self._close_position(ask)
                trade_closed = True
                if self._can_open(ask):
                    self._open_position(1, ask)
                    trade_opened = True

        elif action == self.SELL:
            if self.position == 0:
                if self._can_open(bid):
                    self._open_position(-1, bid)
                    trade_opened = True
            elif self.position == 1 and self._can_close():
                realized_pnl = self._close_position(bid)
                trade_closed = True
                if self._can_open(bid):
                    self._open_position(-1, bid)
                    trade_opened = True

        elif action == self.CLOSE:
            if self.position != 0 and self._can_close():
                close_price = ask if self.position == -1 else bid
                realized_pnl = self._close_position(close_price)
                trade_closed = True

        # ── Update unrealized P/L & equity ─────────

        if self.position == 1:
            self.unrealized_pnl = (bid - self.entry_price) * self.lot_contract
        elif self.position == -1:
            self.unrealized_pnl = (self.entry_price - ask) * self.lot_contract
        else:
            self.unrealized_pnl = 0.0

        self.equity = self.balance + self.unrealized_pnl
        self.free_margin = self.equity - self.margin_used

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Track equity curve
        self.equity_curve.append(self.equity)

        # ══════════════════════════════════════════
        # SIMPLIFIED REWARD (4 components only)
        # ══════════════════════════════════════════

        # 1. Realized P/L — log return (core learning signal)
        if trade_closed and realized_pnl != 0:
            old_bal = self.balance - realized_pnl
            if old_bal > 0 and self.balance > 0:
                reward += np.log(self.balance / old_bal) * 100
            else:
                reward += np.sign(realized_pnl) * 0.1

        # 2. Holding cost — tiny constant per step in position
        if self.position != 0:
            self.steps_in_position += 1
            reward -= 0.0002

        # 3. Spread cost
        if trade_opened and trade_closed:
            reward -= spread * self.lot_contract * 0.02  # Flip = 2x
        elif trade_opened:
            reward -= spread * self.lot_contract * 0.01

        # 4. Drawdown penalty — amplify losses during drawdown
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            if dd > 0.05 and reward < 0:
                reward *= 1.5

        # ── Advance ────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1

        # Force close at end
        if terminated and self.position != 0:
            bar_idx = self.start_idx + min(self.current_step, self.n_steps - 1)
            ep = self.close_prices[bar_idx]
            final_pnl = self._close_position(ep)
            if final_pnl != 0:
                old_b = self.balance - final_pnl
                if old_b > 0 and self.balance > 0:
                    reward += np.log(self.balance / old_b) * 100

        # Margin call / bankruptcy
        if self.equity <= 0 or (self.margin_used > 0 and self.equity < self.margin_used * 0.2):
            terminated = True
            reward -= 10.0
            if self.position != 0:
                bar_idx = self.start_idx + min(self.current_step, self.n_steps - 1)
                self._close_position(self.close_prices[bar_idx])

        obs = self._get_obs_fast()
        return obs, float(reward), terminated, False, self._get_info()

    # ── Observation ────────────────────────────────

    def _get_obs_fast(self):
        """Build observation."""
        idx = self.start_idx + min(self.current_step, self.n_steps - 1)
        self._obs_buffer[:NUM_MARKET_FEATURES] = self.market_features[idx]
        self._obs_buffer[NUM_MARKET_FEATURES] = float(self.position)
        norm_pnl = self.unrealized_pnl / self.initial_balance
        self._obs_buffer[NUM_MARKET_FEATURES + 1] = max(-1.0, min(1.0, norm_pnl))
        return self._obs_buffer.copy()

    def _get_info(self):
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        pf = (self.total_profit / self.total_loss) if self.total_loss > 0 else float("inf")
        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        return {
            "step": self.current_step,
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "margin_used": round(self.margin_used, 2),
            "free_margin": round(self.free_margin, 2),
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
        bar_idx = self.start_idx + min(self.current_step, self.n_steps - 1)
        price = self.close_prices[bar_idx]
        print(
            f"Step {self.current_step:>6d} | Price: {price:>8.2f} | "
            f"Pos: {pos_str:>5s} | P/L: {self.unrealized_pnl:>+8.2f} | "
            f"Bal: {self.balance:>10.2f} | Eq: {self.equity:>10.2f} | "
            f"Trades: {self.total_trades}"
        )


def make_env(df, **kwargs):
    return TradingEnv(df, **kwargs)
