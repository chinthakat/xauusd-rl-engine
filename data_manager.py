"""
Data Manager
=============
Handles fetching XAUUSD M1 data from MT5, loading CSV files,
computing moving averages, and providing a unified data interface.
"""

import os
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import config

logger = logging.getLogger(__name__)


# ── MT5 Data Fetching ──────────────────────────────

def fetch_mt5_rates(symbol=None, timeframe=None, start=None, end=None):
    """
    Fetch historical OHLCV bars from MT5 using copy_rates_range.

    Args:
        symbol:    Instrument name (default: config.SYMBOL)
        timeframe: MT5 timeframe constant (default: mt5.TIMEFRAME_M1)
        start:     datetime — start of range
        end:       datetime — end of range (default: now)

    Returns:
        pd.DataFrame with columns: Open, High, Low, Close, TickVolume, Spread
        Index: datetime
    """
    symbol = symbol or config.SYMBOL
    timeframe = timeframe or mt5.TIMEFRAME_M1
    end = end or datetime.now()
    start = start or (end - timedelta(days=30))

    rates = mt5.copy_rates_range(symbol, timeframe, start, end)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        logger.error(f"Failed to fetch rates for {symbol}: {error}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "TickVolume",
        "real_volume": "RealVolume",
        "spread": "Spread",
    }, inplace=True)

    logger.info(f"Fetched {len(df)} bars for {symbol} from {df.index[0]} to {df.index[-1]}")
    return df


def fetch_latest_bars(symbol=None, timeframe=None, count=200):
    """
    Fetch the most recent N bars using copy_rates_from_pos.

    Args:
        symbol:    Instrument name
        timeframe: MT5 timeframe
        count:     Number of bars to fetch

    Returns:
        pd.DataFrame (same schema as fetch_mt5_rates)
    """
    symbol = symbol or config.SYMBOL
    timeframe = timeframe or mt5.TIMEFRAME_M1

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        logger.error(f"Failed to fetch latest bars for {symbol}: {error}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "TickVolume",
        "real_volume": "RealVolume",
        "spread": "Spread",
    }, inplace=True)

    logger.info(f"Fetched {len(df)} latest bars for {symbol}")
    return df


def get_current_tick(symbol=None):
    """
    Get the latest bid/ask tick for a symbol.

    Returns:
        dict with keys: bid, ask, last, time, spread
        or None on failure
    """
    symbol = symbol or config.SYMBOL
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        logger.error(f"Failed to get tick for {symbol}: {mt5.last_error()}")
        return None

    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last,
        "time": datetime.fromtimestamp(tick.time),
        "spread": round((tick.ask - tick.bid) * 100, 1),  # in points (XAUUSD: 1 point = $0.01)
    }


def get_symbol_info(symbol=None):
    """Get symbol properties (point size, digits, min lot, etc.)."""
    symbol = symbol or config.SYMBOL
    info = mt5.symbol_info(symbol)

    if info is None:
        logger.error(f"Symbol {symbol} not found: {mt5.last_error()}")
        return None

    # Ensure symbol is visible in Market Watch
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)

    return {
        "name": info.name,
        "point": info.point,
        "digits": info.digits,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "trade_contract_size": info.trade_contract_size,
        "spread": info.spread,
        "filling_mode": info.filling_mode,
    }


# ── CSV Data Loading ───────────────────────────────

def load_csv_data(filepath):
    """
    Load XAUUSD M1 data from a CSV file.

    Expected CSV format:
        Date,Time,Open,High,Low,Close,TickVolume,RealVolume,Spread

    Returns:
        pd.DataFrame with datetime index and standard columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Combine Date+Time into datetime index
    if "Date" in df.columns and "Time" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M")
        df.set_index("datetime", inplace=True)
        df.drop(columns=["Date", "Time"], inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

    # Standardize column names
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower == "close":
            rename_map[col] = "Close"
        elif lower in ("tickvolume", "tick_volume"):
            rename_map[col] = "TickVolume"
        elif lower in ("realvolume", "real_volume"):
            rename_map[col] = "RealVolume"
        elif lower == "spread":
            rename_map[col] = "Spread"
    df.rename(columns=rename_map, inplace=True)

    # Sort by time
    df.sort_index(inplace=True)

    logger.info(f"Loaded {len(df)} rows from {os.path.basename(filepath)} "
                f"({df.index[0]} to {df.index[-1]})")
    return df


def load_all_csv_data(data_dir=None):
    """Load and concatenate all XAUUSD CSV files from the data directory."""
    data_dir = data_dir or config.DATA_DIR
    csv_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv") and "XAUUSD" in f.upper()
    ])

    if not csv_files:
        raise FileNotFoundError(f"No XAUUSD CSV files found in {data_dir}")

    frames = [load_csv_data(f) for f in csv_files]
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    logger.info(f"Loaded {len(combined)} total rows from {len(csv_files)} file(s)")
    return combined


# ── Technical Indicators ───────────────────────────

def compute_moving_averages(df, periods=None):
    """
    Add moving average columns to the DataFrame.

    Args:
        df:      DataFrame with 'Close' column
        periods: list of MA periods (default: config.MA_PERIODS)

    Returns:
        DataFrame with added MA columns (e.g., MA5, MA10, MA20, MA100)
    """
    periods = periods or config.MA_PERIODS
    df = df.copy()

    for period in periods:
        col_name = f"MA{period}"
        df[col_name] = df["Close"].rolling(window=period).mean()

    return df


def get_ma_dict(row, periods=None):
    """Extract MA values from a DataFrame row into a dict."""
    periods = periods or config.MA_PERIODS
    return {f"MA{p}": row[f"MA{p}"] for p in periods if f"MA{p}" in row.index}


# ── Data Export ────────────────────────────────────

def save_to_csv(df, filepath):
    """Save DataFrame to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)
    logger.info(f"Saved {len(df)} rows to {filepath}")


# ── Standalone test ────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Testing Data Manager...")
    print("=" * 60)

    # Test CSV loading
    csv_path = os.path.join(config.DATA_DIR, "XAUUSD_M1_2025_03.csv")
    if os.path.exists(csv_path):
        df = load_csv_data(csv_path)
        print(f"\nCSV Data Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date Range: {df.index[0]} → {df.index[-1]}")
        print(f"\nFirst 5 rows:")
        print(df.head())

        # Test MAs
        df = compute_moving_averages(df)
        print(f"\nMA Columns added: {[c for c in df.columns if c.startswith('MA')]}")
        print(f"\nLast 3 rows with MAs:")
        print(df[["Close", "MA5", "MA10", "MA20", "MA100"]].tail(3))

        # Verify MA5 manually
        manual_ma5 = df["Close"].rolling(5).mean()
        match = np.allclose(df["MA5"].dropna(), manual_ma5.dropna(), equal_nan=True)
        print(f"\nMA5 validation: {'✓ PASS' if match else '✗ FAIL'}")
    else:
        print(f"CSV file not found: {csv_path}")

    print("\n" + "=" * 60)
    print("Data Manager test complete.")
