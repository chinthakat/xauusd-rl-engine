"""
Tick Generator
===============
Generates tick sequences from OHLC bars, replicating MT5's tick generation.

Modes:
    - ohlc:       4 ticks per bar (O→H→L→C or O→L→H→C based on direction)
    - open_only:  1 tick per bar (fastest for RL training)
    - every_tick: Interpolated ticks within the bar's range
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Tick:
    """Single price tick."""
    __slots__ = ['timestamp', 'bid', 'ask', 'volume']
    timestamp: float   # Unix timestamp
    bid: float
    ask: float
    volume: float


def generate_ticks_ohlc(timestamp, o, h, l, c, volume, spread):
    """
    MT5 OHLC mode: 4 ticks per bar.
    Direction-aware: bullish bar → O,L,H,C; bearish → O,H,L,C.
    """
    vol_each = volume / 4.0
    half_spread = spread / 2.0

    if c >= o:  # Bullish or doji: O → L → H → C
        prices = [o, l, h, c]
    else:       # Bearish: O → H → L → C
        prices = [o, h, l, c]

    ticks = []
    for i, price in enumerate(prices):
        t = timestamp + i * 15  # Spread across 60s bar
        ticks.append(Tick(
            timestamp=t,
            bid=price - half_spread,
            ask=price + half_spread,
            volume=vol_each,
        ))
    return ticks


def generate_ticks_open_only(timestamp, o, h, l, c, volume, spread):
    """
    Open-only: 1 tick per bar. Fastest mode.
    """
    half_spread = spread / 2.0
    return [Tick(
        timestamp=timestamp,
        bid=o - half_spread,
        ask=o + half_spread,
        volume=volume,
    )]


def generate_ticks_every_tick(timestamp, o, h, l, c, volume, spread, n_ticks=12):
    """
    Every-tick: Interpolated random walk within the bar's range.
    Generates n_ticks ticks that start at Open and end at Close,
    touching High and Low along the way.
    """
    half_spread = spread / 2.0
    vol_each = volume / n_ticks

    # Build path: Open → (random walk touching H and L) → Close
    if n_ticks < 4:
        return generate_ticks_ohlc(timestamp, o, h, l, c, volume, spread)

    prices = np.zeros(n_ticks)
    prices[0] = o
    prices[-1] = c

    # Determine where H and L occur
    if c >= o:  # Bullish
        low_idx = max(1, n_ticks // 4)
        high_idx = max(low_idx + 1, 3 * n_ticks // 4)
    else:       # Bearish
        high_idx = max(1, n_ticks // 4)
        low_idx = max(high_idx + 1, 3 * n_ticks // 4)

    prices[high_idx] = h
    prices[low_idx] = l

    # Interpolate between fixed points
    fixed = sorted([(0, o), (low_idx, l), (high_idx, h), (n_ticks - 1, c)],
                   key=lambda x: x[0])

    for seg_start, seg_end in zip(range(len(fixed) - 1), range(1, len(fixed))):
        i0, p0 = fixed[seg_start]
        i1, p1 = fixed[seg_end]
        n = i1 - i0
        if n <= 1:
            continue
        for j in range(1, n):
            t_frac = j / n
            # Linear interp + small noise
            noise = np.random.uniform(-0.3, 0.3) * (h - l) / n_ticks
            prices[i0 + j] = p0 + (p1 - p0) * t_frac + noise
            prices[i0 + j] = np.clip(prices[i0 + j], l, h)

    ticks = []
    dt = 60.0 / n_ticks  # Spread across 60s
    for i in range(n_ticks):
        ticks.append(Tick(
            timestamp=timestamp + i * dt,
            bid=prices[i] - half_spread,
            ask=prices[i] + half_spread,
            volume=vol_each,
        ))
    return ticks


# ── Factory ──────────────────────────────────────

GENERATORS = {
    "ohlc": generate_ticks_ohlc,
    "open_only": generate_ticks_open_only,
    "every_tick": generate_ticks_every_tick,
}


def generate_ticks(mode, timestamp, o, h, l, c, volume, spread, **kwargs):
    """Generate ticks for a single bar using the specified mode."""
    gen = GENERATORS.get(mode, generate_ticks_ohlc)
    return gen(timestamp, o, h, l, c, volume, spread, **kwargs)
