"""
Tick Data Loader
=================
Loads tick data exported by TickExporter.mq5 from MT5 Strategy Tester.
Converts tick CSV to OHLC DataFrame for the backtester engine,
or provides raw ticks for tick-level backtesting.
"""

import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Default MT5 Common Files path
DEFAULT_COMMON_FILES = os.path.join(
    os.environ.get("APPDATA", ""),
    "MetaQuotes", "Terminal", "Common", "Files"
)


def find_tick_files(folder="tick_export", common_path=None):
    """Find tick data CSV files exported by TickExporter EA."""
    if common_path is None:
        common_path = DEFAULT_COMMON_FILES
    tick_dir = os.path.join(common_path, folder)

    if not os.path.exists(tick_dir):
        logger.warning(f"Tick export directory not found: {tick_dir}")
        return []

    files = [
        os.path.join(tick_dir, f)
        for f in sorted(os.listdir(tick_dir))
        if f.endswith(".csv") and "tick" in f.lower()
    ]
    return files


def load_tick_data(filepath):
    """
    Load raw tick data from TickExporter CSV.

    Returns:
        DataFrame with columns: timestamp, bid, ask, spread_pts, volume,
                                bar_open, bar_high, bar_low, bar_close,
                                ma_fast, ma_medium, ma_slow
    """
    logger.info(f"Loading tick data: {filepath}")

    df = pd.read_csv(filepath)

    # Parse timestamp
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], format="mixed")
    elif "timestamp_ms" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")

    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} ticks from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    return df


def ticks_to_ohlc(tick_df, timeframe_minutes=1):
    """
    Resample tick data into OHLC bars.

    Args:
        tick_df:             Raw tick DataFrame (from load_tick_data)
        timeframe_minutes:   Bar period in minutes (1, 5, 15, 60, etc.)

    Returns:
        DataFrame with OHLC + Volume, DatetimeIndex
    """
    df = tick_df.copy()

    # Use bid as the price reference (standard for charts)
    df = df.set_index("datetime")

    rule = f"{timeframe_minutes}min"

    ohlc = df["bid"].resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    })

    if "volume" in df.columns:
        ohlc["Volume"] = df["volume"].resample(rule).sum()
    elif "volume_real" in df.columns:
        ohlc["Volume"] = df["volume_real"].resample(rule).sum()
    else:
        ohlc["Volume"] = df["bid"].resample(rule).count()

    # Drop bars with no data
    ohlc = ohlc.dropna(subset=["Open"])

    # Add spread (average per bar)
    if "spread_pts" in df.columns:
        ohlc["Spread"] = df["spread_pts"].resample(rule).mean()

    logger.info(f"Resampled to {len(ohlc)} bars ({timeframe_minutes}min)")
    return ohlc


def load_mt5_trades(filepath):
    """
    Load MT5 trade history exported by MACrossoverTest EA.

    Returns:
        DataFrame with trade details for comparison
    """
    logger.info(f"Loading MT5 trades: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} MT5 trade records")
    return df


def compare_results(python_result, mt5_trades_path):
    """
    Compare Python backtester trades with MT5 Strategy Tester trades.

    Prints a side-by-side comparison report.
    """
    mt5_df = load_mt5_trades(mt5_trades_path)

    py_trades = python_result.trades
    py_summary = python_result.summary()

    print("\n" + "=" * 70)
    print("  BACKTESTER COMPARISON: Python vs MT5")
    print("=" * 70)

    # Basic counts
    mt5_count = len(mt5_df)
    py_count = len(py_trades)

    print(f"\n  {'Metric':<30} {'Python':>12} {'MT5':>12} {'Delta':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

    print(f"  {'Total Deals':<30} {py_count:>12d} {mt5_count:>12d} {py_count - mt5_count:>+12d}")

    # P/L comparison
    mt5_total_profit = mt5_df["profit"].sum() if "profit" in mt5_df.columns else 0
    py_total_profit = sum(t.profit for t in py_trades)

    print(f"  {'Total P/L':<30} ${py_total_profit:>11.2f} ${mt5_total_profit:>11.2f} ${py_total_profit - mt5_total_profit:>+11.2f}")

    # Commission
    mt5_commission = mt5_df["commission"].sum() if "commission" in mt5_df.columns else 0
    py_commission = py_summary.get("total_commission", 0)

    print(f"  {'Total Commission':<30} ${py_commission:>11.2f} ${mt5_commission:>11.2f} ${py_commission - mt5_commission:>+11.2f}")

    # Swap
    mt5_swap = mt5_df["swap"].sum() if "swap" in mt5_df.columns else 0
    py_swap = py_summary.get("total_swap", 0)

    print(f"  {'Total Swap':<30} ${py_swap:>11.2f} ${mt5_swap:>11.2f} ${py_swap - mt5_swap:>+11.2f}")

    print(f"\n  {'Net (P/L + Commission + Swap)':<30}")
    mt5_net = mt5_total_profit + mt5_commission + mt5_swap
    py_net = py_total_profit - py_commission + py_swap
    print(f"  {'                              ':<30} ${py_net:>11.2f} ${mt5_net:>11.2f} ${py_net - mt5_net:>+11.2f}")

    print("=" * 70)

    return {
        "py_trades": py_count,
        "mt5_trades": mt5_count,
        "py_profit": py_total_profit,
        "mt5_profit": mt5_total_profit,
        "profit_delta": py_total_profit - mt5_total_profit,
    }
