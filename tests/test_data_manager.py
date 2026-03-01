"""
Unit Tests — Data Manager
===========================
Tests CSV loading, MA computation, and data integrity.
These tests work WITHOUT MT5 — they test against the existing CSV data.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_manager import load_csv_data, compute_moving_averages, load_all_csv_data


# ── Fixtures ─────────────────────────────────────

CSV_PATH = os.path.join(config.DATA_DIR, "XAUUSD_M1_2025_03.csv")

@pytest.fixture
def csv_df():
    """Load the March 2025 CSV data."""
    if not os.path.exists(CSV_PATH):
        pytest.skip(f"CSV file not found: {CSV_PATH}")
    return load_csv_data(CSV_PATH)


@pytest.fixture
def csv_df_with_mas(csv_df):
    """CSV data with moving averages computed."""
    return compute_moving_averages(csv_df)


# ── Test: CSV Loading ────────────────────────────

class TestLoadCSV:

    def test_load_returns_dataframe(self, csv_df):
        assert isinstance(csv_df, pd.DataFrame)
        assert len(csv_df) > 0

    def test_load_shape(self, csv_df):
        """CSV should have ~28K rows and 7 columns."""
        assert len(csv_df) > 20000, f"Expected >20K rows, got {len(csv_df)}"
        assert len(csv_df.columns) >= 6, f"Expected >=6 columns, got {csv_df.columns.tolist()}"

    def test_required_columns(self, csv_df):
        required = {"Open", "High", "Low", "Close"}
        assert required.issubset(set(csv_df.columns)), \
            f"Missing columns: {required - set(csv_df.columns)}"

    def test_datetime_index(self, csv_df):
        assert isinstance(csv_df.index, pd.DatetimeIndex), \
            f"Index type should be DatetimeIndex, got {type(csv_df.index)}"

    def test_data_types(self, csv_df):
        for col in ["Open", "High", "Low", "Close"]:
            assert csv_df[col].dtype in [np.float64, np.float32, float], \
                f"{col} should be float, got {csv_df[col].dtype}"

    def test_no_nan_in_ohlc(self, csv_df):
        for col in ["Open", "High", "Low", "Close"]:
            nan_count = csv_df[col].isna().sum()
            assert nan_count == 0, f"{col} has {nan_count} NaN values"

    def test_timestamps_sorted(self, csv_df):
        assert csv_df.index.is_monotonic_increasing, "Timestamps should be sorted ascending"

    def test_price_range_reasonable(self, csv_df):
        """XAUUSD in March 2025 should be roughly $2800-$3100."""
        assert csv_df["Close"].min() > 2500, f"Min close too low: {csv_df['Close'].min()}"
        assert csv_df["Close"].max() < 3500, f"Max close too high: {csv_df['Close'].max()}"

    def test_high_gte_low(self, csv_df):
        violations = csv_df[csv_df["High"] < csv_df["Low"]]
        assert len(violations) == 0, f"{len(violations)} bars have High < Low"

    def test_close_within_range(self, csv_df):
        """Close should be between Low and High (inclusive)."""
        violations = csv_df[(csv_df["Close"] < csv_df["Low"]) | (csv_df["Close"] > csv_df["High"])]
        assert len(violations) == 0, f"{len(violations)} bars have Close outside High/Low range"

    def test_file_not_found_error(self):
        with pytest.raises(FileNotFoundError):
            load_csv_data("nonexistent_file.csv")


# ── Test: Moving Averages ────────────────────────

class TestMovingAverages:

    def test_ma_columns_added(self, csv_df_with_mas):
        for period in config.MA_PERIODS:
            col = f"MA{period}"
            assert col in csv_df_with_mas.columns, f"Missing column: {col}"

    def test_ma5_values_correct(self, csv_df_with_mas):
        """MA5 should match manual rolling(5).mean()."""
        manual_ma5 = csv_df_with_mas["Close"].rolling(window=5).mean()
        computed_ma5 = csv_df_with_mas["MA5"]

        # Compare only non-NaN values
        mask = manual_ma5.notna()
        assert np.allclose(
            computed_ma5[mask].values,
            manual_ma5[mask].values,
            equal_nan=True
        ), "MA5 values don't match manual computation"

    def test_ma10_values_correct(self, csv_df_with_mas):
        manual_ma10 = csv_df_with_mas["Close"].rolling(window=10).mean()
        computed_ma10 = csv_df_with_mas["MA10"]
        mask = manual_ma10.notna()
        assert np.allclose(computed_ma10[mask].values, manual_ma10[mask].values, equal_nan=True)

    def test_ma100_nan_count(self, csv_df_with_mas):
        """First 99 rows should be NaN for MA100."""
        nan_count = csv_df_with_mas["MA100"].isna().sum()
        assert nan_count == 99, f"MA100 should have 99 NaN values, got {nan_count}"

    def test_ma5_nan_count(self, csv_df_with_mas):
        nan_count = csv_df_with_mas["MA5"].isna().sum()
        assert nan_count == 4, f"MA5 should have 4 NaN values, got {nan_count}"

    def test_ma_values_in_price_range(self, csv_df_with_mas):
        """All MA values should be within the price range."""
        price_min = csv_df_with_mas["Close"].min() - 50
        price_max = csv_df_with_mas["Close"].max() + 50

        for period in config.MA_PERIODS:
            col = f"MA{period}"
            ma_vals = csv_df_with_mas[col].dropna()
            assert ma_vals.min() > price_min, f"{col} min ({ma_vals.min()}) out of range"
            assert ma_vals.max() < price_max, f"{col} max ({ma_vals.max()}) out of range"

    def test_original_df_not_modified(self, csv_df):
        """compute_moving_averages should not modify the original DataFrame."""
        original_cols = set(csv_df.columns)
        _ = compute_moving_averages(csv_df)
        assert set(csv_df.columns) == original_cols, "Original DataFrame was modified"

    def test_custom_periods(self, csv_df):
        """Should support custom MA periods."""
        result = compute_moving_averages(csv_df, periods=[3, 7, 50])
        assert "MA3" in result.columns
        assert "MA7" in result.columns
        assert "MA50" in result.columns


# ── Test: Signal Function ────────────────────────

class TestSignal:

    def test_buy_signal(self):
        """BUY when MA5 > MA20 and MA10 > MA100."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from main import get_signal

        state = {
            "close": 2900.0,
            "ma5": 2905.0,   # above MA20
            "ma10": 2903.0,  # above MA100
            "ma20": 2900.0,
            "ma100": 2895.0,
            "has_position": False,
            "position_type": None,
            "position_profit": 0.0,
        }
        assert get_signal(state) == "BUY"

    def test_sell_signal(self):
        """SELL when MA5 < MA20 and MA10 < MA100."""
        from main import get_signal

        state = {
            "close": 2900.0,
            "ma5": 2895.0,   # below MA20
            "ma10": 2893.0,  # below MA100
            "ma20": 2900.0,
            "ma100": 2905.0,
            "has_position": False,
            "position_type": None,
            "position_profit": 0.0,
        }
        assert get_signal(state) == "SELL"

    def test_close_buy_signal(self):
        """CLOSE BUY when MA5 drops below MA20."""
        from main import get_signal

        state = {
            "close": 2900.0,
            "ma5": 2898.0,   # below MA20
            "ma10": 2903.0,
            "ma20": 2900.0,
            "ma100": 2895.0,
            "has_position": True,
            "position_type": "BUY",
            "position_profit": 10.0,
        }
        assert get_signal(state) == "CLOSE"

    def test_hold_with_nan_mas(self):
        """HOLD when MAs are NaN."""
        from main import get_signal

        state = {
            "close": 2900.0,
            "ma5": float("nan"),
            "ma10": 2903.0,
            "ma20": 2900.0,
            "ma100": 2895.0,
            "has_position": False,
            "position_type": None,
            "position_profit": 0.0,
        }
        assert get_signal(state) == "HOLD"

    def test_hold_when_no_signal(self):
        """HOLD when conditions don't match BUY or SELL."""
        from main import get_signal

        state = {
            "close": 2900.0,
            "ma5": 2901.0,   # above MA20
            "ma10": 2893.0,  # but below MA100 — contradictory
            "ma20": 2900.0,
            "ma100": 2905.0,
            "has_position": False,
            "position_type": None,
            "position_profit": 0.0,
        }
        assert get_signal(state) == "HOLD"


# ── Test: Trade Logger Metrics ───────────────────

class TestMetrics:

    def test_metrics_with_sample_data(self):
        from trade_logger import compute_metrics

        trades = pd.DataFrame({
            "profit": [15.50, -8.20, 22.30, -5.10, 12.70],
        })
        m = compute_metrics(trades)

        assert m["total_trades"] == 5
        assert m["winning_trades"] == 3
        assert m["losing_trades"] == 2
        assert m["win_rate"] == 60.0
        assert m["net_profit"] == round(15.50 - 8.20 + 22.30 - 5.10 + 12.70, 2)
        assert m["max_drawdown"] >= 0

    def test_metrics_empty_data(self):
        from trade_logger import compute_metrics

        m = compute_metrics(pd.DataFrame())
        assert m["total_trades"] == 0
        assert m["win_rate"] == 0.0

    def test_metrics_all_winners(self):
        from trade_logger import compute_metrics

        trades = pd.DataFrame({"profit": [10.0, 20.0, 30.0]})
        m = compute_metrics(trades)
        assert m["win_rate"] == 100.0
        assert m["profit_factor"] == float("inf")

    def test_metrics_all_losers(self):
        from trade_logger import compute_metrics

        trades = pd.DataFrame({"profit": [-10.0, -20.0, -30.0]})
        m = compute_metrics(trades)
        assert m["win_rate"] == 0.0
        assert m["gross_profit"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
