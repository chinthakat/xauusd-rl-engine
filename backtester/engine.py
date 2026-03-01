"""
Backtester Engine
==================
Core backtest loop — steps through historical data bar by bar,
generates ticks, executes strategy signals, and tracks results.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtester.config import BacktestConfig
from backtester.account import Account
from backtester.order_manager import OrderManager
from backtester.tick_generator import generate_ticks

logger = logging.getLogger(__name__)

# Add parent to path for data_manager import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BacktestResult:
    """Container for backtest output."""

    def __init__(self):
        self.config: Optional[BacktestConfig] = None
        self.account: Optional[Account] = None
        self.orders: Optional[OrderManager] = None
        self.ohlc_data: Optional[pd.DataFrame] = None
        self.trade_markers = []   # (timestamp, price, type, ticket)
        self.duration_sec = 0.0
        self.total_bars = 0

    @property
    def trades(self):
        return self.orders.trade_history if self.orders else []

    @property
    def equity_curve(self):
        return self.account.equity_history if self.account else []

    def summary(self) -> dict:
        """Full performance summary."""
        acct = self.account.summary() if self.account else {}
        trades = self.trades

        wins = [t for t in trades if t.profit > 0]
        losses = [t for t in trades if t.profit < 0]

        total_profit = sum(t.profit for t in wins)
        total_loss = abs(sum(t.profit for t in losses))

        acct.update({
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "breakeven_trades": len(trades) - len(wins) - len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0,
            "profit_factor": round(total_profit / total_loss, 4) if total_loss > 0 else float("inf"),
            "avg_profit": round(total_profit / len(wins), 2) if wins else 0,
            "avg_loss": round(-total_loss / len(losses), 2) if losses else 0,
            "largest_win": round(max((t.profit for t in wins), default=0), 2),
            "largest_loss": round(min((t.profit for t in losses), default=0), 2),
            "total_bars": self.total_bars,
            "duration_sec": round(self.duration_sec, 2),
            "bars_per_sec": round(self.total_bars / self.duration_sec, 0) if self.duration_sec > 0 else 0,
        })

        # Sharpe ratio (approximate, using per-trade returns)
        if len(trades) > 1:
            returns = [t.profit for t in trades]
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            acct["sharpe_ratio"] = round(mean_r / std_r * np.sqrt(252), 4) if std_r > 0 else 0
        else:
            acct["sharpe_ratio"] = 0

        return acct


class BacktestEngine:
    """
    Main backtester engine.

    Usage:
        engine = BacktestEngine(config)
        result = engine.run(df, signal_fn)
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.account = Account(self.config)
        self.orders = OrderManager(self.config, self.account)

    def run(self, df: pd.DataFrame, signal_fn: Callable,
            progress: bool = True) -> BacktestResult:
        """
        Run the backtest.

        Args:
            df:         DataFrame with columns: Open, High, Low, Close, Volume
                        and MAs: MA5, MA10, MA20, MA100 (DatetimeIndex)
            signal_fn:  Function(state_dict) → "BUY"|"SELL"|"CLOSE"|"HOLD"
            progress:   Show progress bar

        Returns:
            BacktestResult with trades, equity curve, and summary
        """
        start_time = time.time()
        n = len(df)

        logger.info(f"Starting backtest: {n} bars, tick_mode={self.config.tick_mode}")
        logger.info(f"Config: spread={self.config.spread_points}pts, "
                     f"slippage={self.config.slippage_enabled}, "
                     f"commission={self.config.commission_per_lot}$/lot")

        result = BacktestResult()
        result.config = self.config
        result.account = self.account
        result.orders = self.orders
        result.ohlc_data = df
        result.total_bars = n

        last_day = None

        for i in range(n):
            row = df.iloc[i]
            ts = row.name.timestamp() if hasattr(row.name, 'timestamp') else float(i)

            o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
            vol = row.get("Volume", 0)

            # Determine spread for this bar
            spread = self._get_spread(row)

            # Generate ticks for this bar
            ticks = generate_ticks(
                self.config.tick_mode, ts, o, h, l, c, vol, spread
            )

            # Process each tick
            for tick in ticks:
                # Check SL/TP
                sl_tp_trades = self.orders.check_sl_tp(tick.bid, tick.ask, tick.timestamp)
                for t in sl_tp_trades:
                    result.trade_markers.append(
                        (tick.timestamp, t.exit_price, "close_" + t.type.lower(), t.ticket)
                    )

                # Check stop-out
                if self.orders.has_position:
                    unrealized = self.orders.unrealized_pnl(tick.bid, tick.ask)
                    self.account.update_equity(unrealized)
                    if self.account.check_stop_out():
                        for trade in self.orders.close_all(tick.bid, tick.ask, tick.timestamp):
                            result.trade_markers.append(
                                (tick.timestamp, trade.exit_price, "stop_out", trade.ticket)
                            )
                            logger.warning(f"STOP OUT at bar {i}: equity={self.account.equity:.2f}")

            # Use last tick of bar for signal
            last_tick = ticks[-1]

            # Apply daily swap at rollover
            if hasattr(row.name, 'date'):
                current_day = row.name.date()
                if last_day is not None and current_day != last_day:
                    if row.name.hour == self.config.swap_rollover_hour:
                        self.orders.apply_daily_swap(ts)
                last_day = current_day

            # Build state for signal function
            state = self._build_state(row, last_tick)

            # Get signal
            action = signal_fn(state)

            # Execute action
            if action == "BUY" and not self.orders.has_position:
                pos = self.orders.open_buy(last_tick.ask, ts)
                if pos:
                    result.trade_markers.append((ts, pos.entry_price, "buy", pos.ticket))

            elif action == "SELL" and not self.orders.has_position:
                pos = self.orders.open_sell(last_tick.bid, ts)
                if pos:
                    result.trade_markers.append((ts, pos.entry_price, "sell", pos.ticket))

            elif action == "CLOSE" and self.orders.has_position:
                for pos in list(self.orders.positions):
                    close_price = last_tick.bid if pos.type == 1 else last_tick.ask
                    trade = self.orders.close_position(pos, close_price, ts, "Signal Close")
                    trade.duration_bars = i  # Will be refined
                    result.trade_markers.append(
                        (ts, trade.exit_price, "close_" + trade.type.lower(), trade.ticket)
                    )

            # Update equity
            if self.orders.has_position:
                unrealized = self.orders.unrealized_pnl(last_tick.bid, last_tick.ask)
                self.account.update_equity(unrealized)
            else:
                self.account.update_equity(0.0)

            # Snapshot every N bars
            if i % max(1, n // 500) == 0:
                self.account.snapshot(ts)

            # Progress
            if progress and i % max(1, n // 20) == 0:
                pct = (i + 1) / n * 100
                trades_count = len(self.orders.trade_history)
                net = self.account.balance - self.config.initial_balance
                print(f"\r  [{pct:5.1f}%] Bar {i+1:>6,}/{n:,} | "
                      f"Trades: {trades_count:>4d} | "
                      f"Balance: ${self.account.balance:>10,.2f} | "
                      f"Net P/L: ${net:>+8.2f}", end="", flush=True)

        # Close any remaining positions at end
        if self.orders.has_position:
            last_row = df.iloc[-1]
            spread = self._get_spread(last_row)
            half_s = spread / 2
            final_bid = last_row["Close"] - half_s
            final_ask = last_row["Close"] + half_s
            ts = last_row.name.timestamp() if hasattr(last_row.name, 'timestamp') else float(n)
            for trade in self.orders.close_all(final_bid, final_ask, ts):
                result.trade_markers.append(
                    (ts, trade.exit_price, "close_eod", trade.ticket)
                )

        # Final snapshot
        self.account.update_equity(0.0)
        if hasattr(df.index[-1], 'timestamp'):
            self.account.snapshot(df.index[-1].timestamp())

        result.duration_sec = time.time() - start_time

        if progress:
            print()  # Newline after progress bar

        return result

    def _get_spread(self, row) -> float:
        """Get spread in price units for a bar."""
        if not self.config.spread_enabled:
            return 0.0

        if self.config.spread_mode == "from_data" and "Spread" in row.index:
            return row["Spread"] * self.config.point
        elif self.config.spread_mode == "variable":
            import random
            lo, hi = self.config.spread_variable_range
            return random.randint(lo, hi) * self.config.point
        else:  # fixed
            return self.config.spread_points * self.config.point

    def _build_state(self, row, tick) -> dict:
        """Build state dict for the signal function."""
        pos_type = None
        if self.orders.has_position:
            pos_type = "BUY" if self.orders.position_type == 1 else "SELL"

        return {
            "close": row["Close"],
            "open": row["Open"],
            "high": row["High"],
            "low": row["Low"],
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid,
            "ma5": row.get("MA5", None),
            "ma10": row.get("MA10", None),
            "ma20": row.get("MA20", None),
            "ma100": row.get("MA100", None),
            "has_position": self.orders.has_position,
            "position_type": pos_type,
            "position_profit": self.orders.unrealized_pnl(tick.bid, tick.ask),
            "balance": self.account.balance,
            "equity": self.account.equity,
        }
