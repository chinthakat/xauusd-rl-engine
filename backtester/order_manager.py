"""
Order Manager
==============
Handles order execution with spread, slippage, and latency simulation.
Tracks positions, SL/TP, and trade history.
"""

import random
import logging
from dataclasses import dataclass, field
from typing import Optional, List

from backtester.config import BacktestConfig
from backtester.account import Account

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open position."""
    ticket: int
    type: int           # 1=long, -1=short
    entry_price: float
    lots: float
    sl: float = 0.0     # Stop loss price
    tp: float = 0.0     # Take profit price
    open_time: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    comment: str = ""


@dataclass
class TradeRecord:
    """Completed trade for history."""
    ticket: int
    type: str            # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    lots: float
    profit: float
    commission: float
    swap: float
    slippage: float
    open_time: float
    close_time: float
    duration_bars: int
    comment: str = ""


class OrderManager:
    """
    Manages order execution with realistic market simulation.
    """

    def __init__(self, config: BacktestConfig, account: Account):
        self.config = config
        self.account = account
        self.positions: List[Position] = []
        self.trade_history: List[TradeRecord] = []
        self._next_ticket = 1000
        self._rng = random.Random(42)

    @property
    def has_position(self) -> bool:
        return len(self.positions) > 0

    @property
    def position_type(self) -> Optional[int]:
        if self.positions:
            return self.positions[0].type
        return None

    def open_buy(self, ask_price: float, timestamp: float,
                 sl: float = 0.0, tp: float = 0.0, comment: str = "") -> Optional[Position]:
        """Open a long position at the ask price."""
        if len(self.positions) >= self.config.max_positions:
            return None

        fill_price = self._apply_slippage(ask_price, 1)

        if not self.account.can_open_position(fill_price):
            logger.debug(f"Insufficient margin for BUY @ {fill_price:.2f}")
            return None

        # Commission
        commission = self.account.apply_commission(self.config.lot_size)

        # Lock margin
        self.account.lock_margin(fill_price)

        pos = Position(
            ticket=self._next_ticket,
            type=1,
            entry_price=fill_price,
            lots=self.config.lot_size,
            sl=sl, tp=tp,
            open_time=timestamp,
            commission=commission,
            comment=comment,
        )
        self._next_ticket += 1
        self.positions.append(pos)
        return pos

    def open_sell(self, bid_price: float, timestamp: float,
                  sl: float = 0.0, tp: float = 0.0, comment: str = "") -> Optional[Position]:
        """Open a short position at the bid price."""
        if len(self.positions) >= self.config.max_positions:
            return None

        fill_price = self._apply_slippage(bid_price, -1)

        if not self.account.can_open_position(fill_price):
            logger.debug(f"Insufficient margin for SELL @ {fill_price:.2f}")
            return None

        commission = self.account.apply_commission(self.config.lot_size)
        self.account.lock_margin(fill_price)

        pos = Position(
            ticket=self._next_ticket,
            type=-1,
            entry_price=fill_price,
            lots=self.config.lot_size,
            sl=sl, tp=tp,
            open_time=timestamp,
            commission=commission,
            comment=comment,
        )
        self._next_ticket += 1
        self.positions.append(pos)
        return pos

    def close_position(self, pos: Position, close_price: float,
                       timestamp: float, comment: str = "") -> TradeRecord:
        """Close a position and record the trade."""
        # Apply slippage to close price
        if pos.type == 1:  # Long: close at bid
            fill_price = self._apply_slippage(close_price, -1)
        else:              # Short: close at ask
            fill_price = self._apply_slippage(close_price, 1)

        # Calculate P/L
        if pos.type == 1:
            pnl = (fill_price - pos.entry_price) * pos.lots * self.config.contract_size
        else:
            pnl = (pos.entry_price - fill_price) * pos.lots * self.config.contract_size

        # Close commission
        close_commission = self.account.apply_commission(pos.lots)

        # Apply to account
        self.account.apply_realized_pnl(pnl)
        self.account.release_margin(pos.entry_price)

        # Slippage cost tracking
        slippage = abs(fill_price - close_price) * pos.lots * self.config.contract_size
        self.account.total_slippage_cost += slippage

        # Record trade
        trade = TradeRecord(
            ticket=pos.ticket,
            type="BUY" if pos.type == 1 else "SELL",
            entry_price=pos.entry_price,
            exit_price=fill_price,
            lots=pos.lots,
            profit=round(pnl, 2),
            commission=round(pos.commission + close_commission, 2),
            swap=round(pos.swap, 2),
            slippage=round(slippage, 4),
            open_time=pos.open_time,
            close_time=timestamp,
            duration_bars=0,  # Set by engine
            comment=comment or pos.comment,
        )
        self.trade_history.append(trade)

        # Remove position
        self.positions = [p for p in self.positions if p.ticket != pos.ticket]
        return trade

    def close_all(self, bid: float, ask: float, timestamp: float) -> List[TradeRecord]:
        """Close all open positions."""
        trades = []
        for pos in list(self.positions):
            close_price = bid if pos.type == 1 else ask
            trades.append(self.close_position(pos, close_price, timestamp, "Close All"))
        return trades

    def check_sl_tp(self, bid: float, ask: float, timestamp: float) -> List[TradeRecord]:
        """
        Check if any position's SL or TP has been hit.
        Returns list of trades closed by SL/TP.
        """
        trades = []
        for pos in list(self.positions):
            close_price = None
            comment = ""

            if pos.type == 1:  # Long position — check against bid
                if pos.sl > 0 and bid <= pos.sl:
                    close_price = pos.sl
                    comment = "SL Hit"
                elif pos.tp > 0 and bid >= pos.tp:
                    close_price = pos.tp
                    comment = "TP Hit"
            else:              # Short position — check against ask
                if pos.sl > 0 and ask >= pos.sl:
                    close_price = pos.sl
                    comment = "SL Hit"
                elif pos.tp > 0 and ask <= pos.tp:
                    close_price = pos.tp
                    comment = "TP Hit"

            if close_price is not None:
                trades.append(self.close_position(pos, close_price, timestamp, comment))

        return trades

    def unrealized_pnl(self, bid: float, ask: float) -> float:
        """Calculate total unrealized P/L of all positions."""
        pnl = 0.0
        for pos in self.positions:
            if pos.type == 1:  # Long
                pnl += (bid - pos.entry_price) * pos.lots * self.config.contract_size
            else:              # Short
                pnl += (pos.entry_price - ask) * pos.lots * self.config.contract_size
        return pnl

    def apply_daily_swap(self, timestamp: float):
        """Apply overnight swap to all open positions."""
        for pos in self.positions:
            day = 1
            # Triple swap on configured day (default Wednesday)
            # Simple check: if day-of-week matches
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            if dt.weekday() == self.config.swap_triple_day:
                day = 3

            swap = self.account.apply_swap(pos.type, pos.lots, day)
            pos.swap += swap

    def _apply_slippage(self, price: float, direction: int) -> float:
        """
        Apply slippage to a price.

        direction: 1 for buy-side (price goes up), -1 for sell-side (price goes down)
        """
        if not self.config.slippage_enabled:
            return price

        min_slip, max_slip = self.config.slippage_points
        slip_points = self._rng.randint(min_slip, max_slip)
        slippage = slip_points * self.config.point * direction
        return price + slippage
