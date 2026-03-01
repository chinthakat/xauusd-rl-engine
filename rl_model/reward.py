"""
Reward Function
================
Multi-component reward for the RL trading agent.
Balances profitability, risk management, and trading costs.
"""

import numpy as np


class RewardCalculator:
    """
    Computes step rewards for the trading environment.

    Components:
        1. Realized P/L (log return on trade close)
        2. Unrealized P/L change (encourages profitable direction)
        3. Holding penalty (discourages idle exposure)
        4. Trade cost (spread penalty on open/close)
        5. Drawdown penalty (extra punishment when in deep drawdown)
    """

    def __init__(
        self,
        spread_cost=0.25,         # Spread in dollars per unit (25 points × $0.01)
        holding_penalty=0.0001,   # Per-step penalty while in a position
        drawdown_threshold=0.05,  # 5% drawdown triggers extra penalty
        drawdown_multiplier=2.0,  # Multiply negative reward by this during drawdown
        lot_size=0.01,
        contract_size=100,        # XAUUSD: 1 lot = 100 oz
    ):
        self.spread_cost = spread_cost
        self.holding_penalty = holding_penalty
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_multiplier = drawdown_multiplier
        self.lot_size = lot_size
        self.contract_size = contract_size

        # State tracking
        self.peak_balance = 0.0
        self.initial_balance = 0.0

    def reset(self, initial_balance):
        """Reset for new episode."""
        self.peak_balance = initial_balance
        self.initial_balance = initial_balance

    def compute(
        self,
        action,
        realized_pnl=0.0,
        unrealized_pnl_change=0.0,
        current_balance=0.0,
        has_position=False,
        trade_opened=False,
        trade_closed=False,
    ):
        """
        Compute the reward for the current step.

        Args:
            action:                 The action taken (0=hold, 1=buy, 2=sell, 3=close)
            realized_pnl:          Dollar P/L from closing a trade (0 if no close)
            unrealized_pnl_change: Change in unrealized P/L since last step
            current_balance:       Current account balance
            has_position:          Whether agent is currently in a position
            trade_opened:          Whether a new trade was opened this step
            trade_closed:          Whether a trade was closed this step

        Returns:
            float: total reward for this step
        """
        reward = 0.0

        # ── 1. Realized P/L (logarithmic for risk sensitivity) ──
        if trade_closed and realized_pnl != 0:
            # Log return: penalizes large losses more than it rewards gains
            old_balance = current_balance - realized_pnl
            if old_balance > 0 and current_balance > 0:
                reward += np.log(current_balance / old_balance) * 100  # Scale up
            else:
                reward += np.sign(realized_pnl) * 0.1

        # ── 2. Unrealized P/L change (directional encouragement) ──
        if has_position:
            reward += unrealized_pnl_change * 0.01  # Small weight

        # ── 3. Holding penalty ──
        if has_position:
            reward -= self.holding_penalty

        # ── 4. Trade cost (spread) ──
        if trade_opened or trade_closed:
            cost = self.spread_cost * self.lot_size * self.contract_size
            reward -= cost * 0.01  # Normalize: $0.25 → -0.0025

        # ── 5. Drawdown penalty ──
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        if self.peak_balance > 0:
            drawdown = (self.peak_balance - current_balance) / self.peak_balance
            if drawdown > self.drawdown_threshold:
                # Extra penalty proportional to drawdown severity
                if reward < 0:
                    reward *= self.drawdown_multiplier

        return float(reward)
