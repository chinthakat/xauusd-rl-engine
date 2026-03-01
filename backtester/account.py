"""
Account Manager
================
Tracks balance, equity, margin, and handles swap/commission.
"""

import logging
from backtester.config import BacktestConfig

logger = logging.getLogger(__name__)


class Account:
    """Trading account with margin and cost tracking."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.used_margin = 0.0
        self.free_margin = config.initial_balance

        # Stats
        self.peak_balance = config.initial_balance
        self.peak_equity = config.initial_balance
        self.total_commission = 0.0
        self.total_swap = 0.0
        self.total_slippage_cost = 0.0
        self.margin_call_count = 0
        self.stop_out_count = 0

        # Snapshots for equity curve
        self.equity_history = []
        self.balance_history = []

    def update_equity(self, unrealized_pnl: float):
        """Update equity based on unrealized P/L."""
        self.equity = self.balance + unrealized_pnl
        self.free_margin = self.equity - self.used_margin

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def apply_realized_pnl(self, pnl: float):
        """Apply realized profit/loss to balance."""
        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

    def apply_commission(self, lots: float, sides: int = 1) -> float:
        """Deduct commission. Returns commission amount."""
        if not self.config.commission_enabled:
            return 0.0
        commission = self.config.commission_per_lot * lots * sides
        self.balance -= commission
        self.total_commission += commission
        return commission

    def apply_swap(self, position_type: int, lots: float, days: int = 1) -> float:
        """
        Apply overnight swap.

        Args:
            position_type: 1=long, -1=short
            lots: Position size in lots
            days: Number of days (3 on triple-swap day)
        """
        if not self.config.swap_enabled:
            return 0.0

        rate = self.config.swap_long if position_type == 1 else self.config.swap_short
        swap = rate * lots * days
        self.balance += swap  # Swap can be positive (credit) or negative (debit)
        self.total_swap += swap
        return swap

    def calculate_margin(self, price: float) -> float:
        """Calculate margin required for one position."""
        return self.config.margin_required(price)

    def lock_margin(self, price: float):
        """Reserve margin for a new position."""
        margin = self.calculate_margin(price)
        self.used_margin += margin
        self.free_margin = self.equity - self.used_margin

    def release_margin(self, price: float):
        """Release margin after closing a position."""
        margin = self.calculate_margin(price)
        self.used_margin = max(0, self.used_margin - margin)
        self.free_margin = self.equity - self.used_margin

    def can_open_position(self, price: float) -> bool:
        """Check if there's enough margin to open a position."""
        if not self.config.margin_enabled:
            return self.balance > 0
        margin = self.calculate_margin(price)
        return self.free_margin >= margin

    def check_margin_call(self) -> bool:
        """Check if margin call level is breached."""
        if not self.config.margin_enabled or self.used_margin <= 0:
            return False
        level = (self.equity / self.used_margin) * 100
        if level <= self.config.margin_call_level:
            self.margin_call_count += 1
            return True
        return False

    def check_stop_out(self) -> bool:
        """Check if stop-out level is breached (forced close)."""
        if not self.config.margin_enabled or self.used_margin <= 0:
            return False
        level = (self.equity / self.used_margin) * 100
        if level <= self.config.stop_out_level:
            self.stop_out_count += 1
            return True
        return False

    def snapshot(self, timestamp):
        """Record a point on the equity/balance curve."""
        self.equity_history.append({
            "timestamp": timestamp,
            "equity": round(self.equity, 2),
            "balance": round(self.balance, 2),
            "used_margin": round(self.used_margin, 2),
            "free_margin": round(self.free_margin, 2),
        })

    def drawdown(self) -> float:
        """Current drawdown from peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def max_drawdown(self) -> float:
        """Maximum drawdown from equity history."""
        if not self.equity_history:
            return 0.0
        peak = 0.0
        max_dd = 0.0
        for snap in self.equity_history:
            eq = snap["equity"]
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def summary(self) -> dict:
        """Account performance summary."""
        return {
            "initial_balance": self.config.initial_balance,
            "final_balance": round(self.balance, 2),
            "final_equity": round(self.equity, 2),
            "net_profit": round(self.balance - self.config.initial_balance, 2),
            "return_pct": round((self.balance / self.config.initial_balance - 1) * 100, 2),
            "peak_balance": round(self.peak_balance, 2),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 2),
            "total_commission": round(self.total_commission, 2),
            "total_swap": round(self.total_swap, 2),
            "margin_calls": self.margin_call_count,
            "stop_outs": self.stop_out_count,
        }
