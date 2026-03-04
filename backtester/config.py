"""
Backtester Configuration
=========================
All market simulation features are toggleable.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BacktestConfig:
    """
    Configuration for the backtester engine.
    Every feature can be enabled/disabled independently.
    """

    # ── Account ──────────────────────────────
    initial_balance: float = 10000.0
    currency: str = "USD"

    # ── Instrument ───────────────────────────
    symbol: str = "XAUUSD"
    lot_size: float = 0.01
    contract_size: float = 100.0       # 1 lot = 100 oz for XAUUSD
    point: float = 0.01               # Minimum price increment
    digits: int = 2                    # Price decimal places

    # ── Spread ───────────────────────────────
    spread_enabled: bool = True
    spread_mode: str = "fixed"         # "fixed", "variable", "from_data"
    spread_points: int = 25            # 25 points = $0.25 for XAUUSD
    spread_variable_range: Tuple[int, int] = (15, 40)  # Min/max for variable

    # ── Slippage ─────────────────────────────
    slippage_enabled: bool = True
    slippage_points: Tuple[int, int] = (0, 5)  # Random range in points

    # ── Latency ──────────────────────────────
    latency_enabled: bool = True
    latency_ms: Tuple[int, int] = (10, 50)     # Simulated network delay

    # ── Commission ───────────────────────────
    commission_enabled: bool = True
    commission_per_lot: float = 7.0    # $ per lot per side
    commission_mode: str = "per_lot"   # "per_lot" or "per_trade"

    # ── Swap (overnight interest) ────────────
    swap_enabled: bool = True
    swap_long: float = -3.50           # $/lot/day for long positions
    swap_short: float = 1.20           # $/lot/day for short positions
    swap_rollover_hour: int = 0        # Hour when swap is applied (server time)
    swap_triple_day: int = 2           # Wednesday=2 (triple swap day)

    # ── Margin ───────────────────────────────
    margin_enabled: bool = True
    leverage: int = 100
    margin_call_level: float = 50.0    # % equity/used_margin
    stop_out_level: float = 20.0       # % forced close level

    # ── Tick Generation ──────────────────────
    tick_mode: str = "ohlc"            # "ohlc", "open_only", "every_tick"
    ticks_per_bar: int = 12            # For 'every_tick' mode

    # ── Position Rules ───────────────────────
    max_positions: int = 1             # Max simultaneous positions
    position_mode: str = "netting"     # "netting" or "hedging"

    # ── Risk Controls ────────────────────────
    max_lot_size: float = 10.0
    min_lot_size: float = 0.01
    lot_step: float = 0.01

    def spread_dollars(self) -> float:
        """Spread in dollars."""
        return self.spread_points * self.point

    def trade_cost(self) -> float:
        """Total cost per round-trip trade (spread + commission)."""
        cost = 0.0
        if self.spread_enabled:
            cost += self.spread_dollars() * self.lot_size * self.contract_size
        if self.commission_enabled:
            cost += self.commission_per_lot * self.lot_size * 2  # Both sides
        return cost

    def margin_required(self, price: float) -> float:
        """Margin required to open a position at given price."""
        if not self.margin_enabled:
            return 0.0
        return (price * self.lot_size * self.contract_size) / self.leverage

    def to_dict(self) -> dict:
        """Export config as dict for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ── Presets ──────────────────────────────────────

def ideal_config() -> BacktestConfig:
    """No costs, no delays — for baseline comparison."""
    return BacktestConfig(
        spread_enabled=False,
        slippage_enabled=False,
        latency_enabled=False,
        commission_enabled=False,
        swap_enabled=False,
        margin_enabled=False,
    )


def realistic_config() -> BacktestConfig:
    """Standard realistic settings."""
    return BacktestConfig()


def harsh_config() -> BacktestConfig:
    """Worst-case conditions — stress test."""
    return BacktestConfig(
        spread_points=50,
        slippage_points=(2, 10),
        latency_ms=(50, 200),
        commission_per_lot=10.0,
        swap_long=-5.0,
        swap_short=-1.0,
        leverage=50,
    )
