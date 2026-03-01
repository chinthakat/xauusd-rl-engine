"""
Trade Logger & Performance Metrics
====================================
Logs trades and account snapshots to CSV files.
Computes performance metrics: win rate, profit factor, max drawdown, Sharpe ratio.
"""

import os
import logging
import threading
from datetime import datetime

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Thread lock for file writes
_file_lock = threading.Lock()


# ── Ensure log directory ─────────────────────────

def _ensure_log_dir():
    os.makedirs(config.LOG_DIR, exist_ok=True)


# ── Trade Logging ────────────────────────────────

def log_trade(trade_data):
    """
    Append a trade entry to the trade log CSV.

    Args:
        trade_data: dict from OrderResult.to_dict() or equivalent
    """
    _ensure_log_dir()

    df_row = pd.DataFrame([trade_data])

    with _file_lock:
        if os.path.exists(config.TRADE_LOG_FILE):
            df_row.to_csv(config.TRADE_LOG_FILE, mode="a", header=False, index=False)
        else:
            df_row.to_csv(config.TRADE_LOG_FILE, mode="w", header=True, index=False)

    logger.debug(f"Logged trade: {trade_data.get('action')} {trade_data.get('symbol')}")


def log_account_snapshot(account_info):
    """
    Append an account snapshot to the account log CSV.

    Args:
        account_info: dict from account_monitor.get_account_info()
    """
    _ensure_log_dir()

    df_row = pd.DataFrame([account_info])

    with _file_lock:
        if os.path.exists(config.ACCOUNT_LOG_FILE):
            df_row.to_csv(config.ACCOUNT_LOG_FILE, mode="a", header=False, index=False)
        else:
            df_row.to_csv(config.ACCOUNT_LOG_FILE, mode="w", header=True, index=False)


# ── Load Logs ────────────────────────────────────

def load_trade_log():
    """Load the trade log as a DataFrame."""
    if not os.path.exists(config.TRADE_LOG_FILE):
        return pd.DataFrame()
    return pd.read_csv(config.TRADE_LOG_FILE)


def load_account_log():
    """Load the account log as a DataFrame."""
    if not os.path.exists(config.ACCOUNT_LOG_FILE):
        return pd.DataFrame()
    return pd.read_csv(config.ACCOUNT_LOG_FILE)


# ── Performance Metrics ──────────────────────────

def compute_metrics(trade_log_df=None):
    """
    Compute performance metrics from the trade log.

    Returns:
        dict with: total_trades, winning_trades, win_rate, gross_profit,
        gross_loss, net_profit, profit_factor, max_drawdown, avg_trade_profit,
        sharpe_ratio
    """
    if trade_log_df is None:
        trade_log_df = load_trade_log()

    if trade_log_df.empty or "profit" not in trade_log_df.columns:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_profit": 0.0,
            "sharpe_ratio": 0.0,
        }

    # Filter only closed trades with profit data
    closed = trade_log_df[trade_log_df["profit"].notna()].copy()
    if closed.empty:
        return compute_metrics(pd.DataFrame())  # returns zeros

    profits = closed["profit"].astype(float)

    total_trades = len(closed)
    winners = profits[profits > 0]
    losers = profits[profits < 0]

    gross_profit = winners.sum() if len(winners) > 0 else 0.0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
    net_profit = profits.sum()
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_trade_profit = profits.mean()

    # Max drawdown from cumulative profit curve
    cum_profit = profits.cumsum()
    running_max = cum_profit.cummax()
    drawdown = running_max - cum_profit
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0

    # Sharpe ratio (annualized for daily, but we compute per-trade here)
    sharpe_ratio = 0.0
    if profits.std() > 0:
        sharpe_ratio = (profits.mean() / profits.std()) * np.sqrt(252)  # annualized

    return {
        "total_trades": total_trades,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": round(win_rate, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_profit": round(net_profit, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_drawdown, 2),
        "avg_trade_profit": round(avg_trade_profit, 2),
        "sharpe_ratio": round(sharpe_ratio, 4),
    }


def print_summary(trade_log_df=None):
    """Print a formatted performance summary to console."""
    metrics = compute_metrics(trade_log_df)

    print("\n" + "=" * 50)
    print("      TRADING PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Total Trades:     {metrics['total_trades']}")
    print(f"  Winning Trades:   {metrics['winning_trades']}")
    print(f"  Losing Trades:    {metrics.get('losing_trades', 0)}")
    print(f"  Win Rate:         {metrics['win_rate']}%")
    print(f"  ──────────────────────────────")
    print(f"  Gross Profit:     ${metrics['gross_profit']:.2f}")
    print(f"  Gross Loss:       ${metrics['gross_loss']:.2f}")
    print(f"  Net Profit:       ${metrics['net_profit']:.2f}")
    print(f"  Profit Factor:    {metrics['profit_factor']}")
    print(f"  ──────────────────────────────")
    print(f"  Avg Trade P/L:    ${metrics['avg_trade_profit']:.2f}")
    print(f"  Max Drawdown:     ${metrics['max_drawdown']:.2f}")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']}")
    print("=" * 50)

    return metrics


# ── Standalone test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Trade Logger — Test")
    print("=" * 50)

    # Test with sample data
    sample_trades = pd.DataFrame({
        "timestamp": [datetime.now().isoformat()] * 10,
        "action": ["BUY", "CLOSE", "SELL", "CLOSE", "BUY", "CLOSE", "SELL", "CLOSE", "BUY", "CLOSE"],
        "symbol": ["XAUUSD"] * 10,
        "profit": [None, 15.50, None, -8.20, None, 22.30, None, -5.10, None, 12.70],
    })

    print("\nSample trade log:")
    print(sample_trades.to_string())

    print_summary(sample_trades)

    print("\nTrade Logger test complete.")
