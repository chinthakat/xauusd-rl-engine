"""
Backtester CLI
===============
Run backtests from the command line.

Usage:
    python -m backtester.run                           # Default: MA crossover
    python -m backtester.run --model rl_ppo            # RL PPO model
    python -m backtester.run --no-spread --no-slippage # Ideal conditions
    python -m backtester.run --preset harsh             # Stress test
    python -m backtester.run --spread 30 --tick-mode every_tick
    python -m backtester.run --no-chart                # Skip visualization
"""

import os
import sys
import argparse
import logging

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import BacktestConfig, ideal_config, realistic_config, harsh_config
from backtester.engine import BacktestEngine
from backtester.report import print_report, print_trade_log, export_trades_csv
from backtester.visualizer import create_chart
from backtester.tick_data import find_tick_files, load_tick_data, ticks_to_ohlc, compare_results

logger = logging.getLogger(__name__)


def get_signal_fn(model_name):
    """Get the signal function based on model name."""
    if model_name == "rl_ppo":
        from rl_model.live_agent import RLAgent
        agent = RLAgent()
        return agent.get_signal
    else:
        from main import get_signal
        return get_signal


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Industry-Level Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backtester.run                         # MA crossover, realistic
  python -m backtester.run --preset ideal          # Zero costs (baseline)
  python -m backtester.run --preset harsh          # Worst-case stress test
  python -m backtester.run --model rl_ppo          # RL PPO agent
  python -m backtester.run --no-spread --spread 0  # Custom: no spread
  python -m backtester.run --tick-mode every_tick  # High-fidelity
        """,
    )

    # Model
    parser.add_argument("--model", default="ma_crossover",
                        choices=["ma_crossover", "rl_ppo"],
                        help="Trading model to test")

    # Presets
    parser.add_argument("--preset", default="realistic",
                        choices=["ideal", "realistic", "harsh"],
                        help="Configuration preset")

    # Override individual settings
    parser.add_argument("--spread", type=int, default=None, help="Override spread (points)")
    parser.add_argument("--no-spread", action="store_true", help="Disable spread")
    parser.add_argument("--no-slippage", action="store_true", help="Disable slippage")
    parser.add_argument("--no-latency", action="store_true", help="Disable latency")
    parser.add_argument("--no-commission", action="store_true", help="Disable commission")
    parser.add_argument("--no-swap", action="store_true", help="Disable swap")
    parser.add_argument("--no-margin", action="store_true", help="Disable margin")

    parser.add_argument("--tick-mode", default=None,
                        choices=["ohlc", "open_only", "every_tick"],
                        help="Tick generation mode")
    parser.add_argument("--lot-size", type=float, default=None, help="Lot size")
    parser.add_argument("--balance", type=float, default=None, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=None, help="Leverage")

    # Tick data from MT5
    parser.add_argument("--tick-data", type=str, default=None,
                        help="Path to tick CSV from TickExporter EA (uses real MT5 ticks)")
    parser.add_argument("--compare-mt5", type=str, default=None,
                        help="Path to MT5 trade log CSV for comparison")

    # Output
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--export-csv", action="store_true", help="Export trades to CSV")

    args = parser.parse_args()

    # ── Build config ─────────────────────────
    if args.preset == "ideal":
        config = ideal_config()
    elif args.preset == "harsh":
        config = harsh_config()
    else:
        config = realistic_config()

    # Apply overrides
    if args.no_spread:
        config.spread_enabled = False
    if args.spread is not None:
        config.spread_points = args.spread
        config.spread_enabled = args.spread > 0
    if args.no_slippage:
        config.slippage_enabled = False
    if args.no_latency:
        config.latency_enabled = False
    if args.no_commission:
        config.commission_enabled = False
    if args.no_swap:
        config.swap_enabled = False
    if args.no_margin:
        config.margin_enabled = False
    if args.tick_mode:
        config.tick_mode = args.tick_mode
    if args.lot_size:
        config.lot_size = args.lot_size
    if args.balance:
        config.initial_balance = args.balance
    if args.leverage:
        config.leverage = args.leverage

    # ── Load data ────────────────────────────
    print("\n" + "=" * 70)
    print("  INDUSTRY-LEVEL BACKTESTER")
    print("=" * 70)

    import data_manager

    if args.tick_data:
        # Load MT5 tick data and resample to OHLC
        print(f"\n  Loading MT5 tick data: {args.tick_data}")
        raw_ticks = load_tick_data(args.tick_data)
        df = ticks_to_ohlc(raw_ticks, timeframe_minutes=1)
        df = data_manager.compute_moving_averages(df)
        df = df.dropna(subset=["MA100"])
        config.spread_mode = "from_data"  # Use real spreads from ticks
        print(f"  Tick data: {len(raw_ticks):,} ticks → {len(df):,} M1 bars")
    else:
        print(f"\n  Loading CSV data...")
        df = data_manager.load_all_csv_data()
        df = data_manager.compute_moving_averages(df)
        df = df.dropna(subset=["MA100"])

    print(f"  Data: {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    # ── Get signal function ──────────────────
    print(f"  Model: {args.model}")
    signal_fn = get_signal_fn(args.model)

    # ── Run backtest ─────────────────────────
    engine = BacktestEngine(config)
    result = engine.run(df, signal_fn, progress=True)

    # ── Print report ─────────────────────────
    print_report(result)
    print_trade_log(result, max_trades=15)

    # ── Export CSV ────────────────────────────
    if args.export_csv:
        csv_path = os.path.join("logs", "backtest_trades.csv")
        export_trades_csv(result, csv_path)

    # ── MT5 Comparison ───────────────────────
    if args.compare_mt5:
        compare_results(result, args.compare_mt5)

    # ── Interactive chart ────────────────────
    if not args.no_chart:
        print("\n  Generating interactive chart...")
        create_chart(result, show=True)
        print("  Chart opened in browser.")

    print()


if __name__ == "__main__":
    main()
