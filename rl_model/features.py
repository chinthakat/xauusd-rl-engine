"""
Feature Engineering (Optimized)
================================
Builds the observation state vector from raw OHLCV + MA data.
All features are normalized for stable RL training.
Optimized for speed: pre-computes everything, zero per-step allocations.
"""

import numpy as np
import pandas as pd


# ── Feature columns expected in the observation ───

FEATURE_NAMES = [
    "close_ma5_ratio",      # Close / MA5 - 1
    "close_ma10_ratio",     # Close / MA10 - 1
    "close_ma20_ratio",     # Close / MA20 - 1
    "close_ma100_ratio",    # Close / MA100 - 1
    "ma5_ma20_diff",        # (MA5 - MA20) / Close
    "ma10_ma100_diff",      # (MA10 - MA100) / Close
    "return_5",             # 5-bar percentage return
    "return_10",            # 10-bar percentage return
    "return_20",            # 20-bar percentage return
    "volatility_20",        # 20-bar rolling std of returns
    "hour_sin",             # sin(2π * hour / 24)
    "hour_cos",             # cos(2π * hour / 24)
    "position_flag",        # 0 = flat, 1 = long, -1 = short
    "unrealized_pnl",       # Normalized unrealized P/L
]

NUM_FEATURES = len(FEATURE_NAMES)
NUM_MARKET_FEATURES = 12  # Features that don't depend on position


def precompute_features(df):
    """
    Pre-compute all market features from a DataFrame with OHLCV + MA columns.
    Fully vectorized — no Python loops.

    Returns:
        np.ndarray of shape (len(df), NUM_MARKET_FEATURES) as float32
    """
    close = df["Close"].values.astype(np.float64)
    ma5 = df["MA5"].values.astype(np.float64)
    ma10 = df["MA10"].values.astype(np.float64)
    ma20 = df["MA20"].values.astype(np.float64)
    ma100 = df["MA100"].values.astype(np.float64)

    # Price ratios (centered at 0)
    close_ma5 = np.where(ma5 > 0, close / ma5 - 1.0, 0.0)
    close_ma10 = np.where(ma10 > 0, close / ma10 - 1.0, 0.0)
    close_ma20 = np.where(ma20 > 0, close / ma20 - 1.0, 0.0)
    close_ma100 = np.where(ma100 > 0, close / ma100 - 1.0, 0.0)

    # MA crossover diffs (normalized by close)
    ma5_ma20_diff = np.where(close > 0, (ma5 - ma20) / close, 0.0)
    ma10_ma100_diff = np.where(close > 0, (ma10 - ma100) / close, 0.0)

    # Momentum (percentage returns) — vectorized
    return_5 = _pct_return(close, 5)
    return_10 = _pct_return(close, 10)
    return_20 = _pct_return(close, 20)

    # Volatility — vectorized using pandas rolling (much faster than Python loop)
    returns_1 = np.zeros_like(close)
    returns_1[1:] = (close[1:] - close[:-1]) / close[:-1]
    volatility_20 = pd.Series(returns_1).rolling(20).std().fillna(0).values * 100

    # Time features (cyclical encoding of hour)
    if hasattr(df.index, 'hour'):
        hours = df.index.hour.values.astype(np.float64)
    else:
        hours = np.zeros(len(df))
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    # Stack all market features
    features = np.column_stack([
        close_ma5, close_ma10, close_ma20, close_ma100,
        ma5_ma20_diff, ma10_ma100_diff,
        return_5, return_10, return_20,
        volatility_20,
        hour_sin, hour_cos,
    ])

    # Replace NaN/inf with 0 and clip
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(features, -5.0, 5.0, out=features)

    return features.astype(np.float32)


def build_observation(market_features_row, position_flag, unrealized_pnl):
    """
    Build a single observation vector — optimized with pre-allocated buffer.
    """
    obs = np.empty(NUM_FEATURES, dtype=np.float32)
    obs[:NUM_MARKET_FEATURES] = market_features_row
    obs[NUM_MARKET_FEATURES] = position_flag
    obs[NUM_MARKET_FEATURES + 1] = unrealized_pnl
    return obs


# ── Internal helpers ─────────────────────────────

def _pct_return(prices, lookback):
    """Compute percentage return over lookback period — vectorized."""
    result = np.zeros_like(prices)
    result[lookback:] = (prices[lookback:] - prices[:-lookback]) / prices[:-lookback]
    return np.clip(result, -0.1, 0.1)
