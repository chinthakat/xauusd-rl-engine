"""
Main Orchestrator — MT5 Trade Execution System
================================================
Ties all modules together into a live trading loop.

Supports two signal models:
    - ma_crossover: Simple MA crossover baseline (default)
    - rl_ppo:       Trained PPO reinforcement learning agent

Usage:
    python main.py                              # Live mode, MA crossover
    python main.py --model rl_ppo               # Live mode, RL agent
    python main.py --backtest                   # Backtest, MA crossover
    python main.py --backtest --model rl_ppo    # Backtest, RL agent
"""

import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime

import pandas as pd

import config

# ── Logging Setup ────────────────────────────────

def setup_logging():
    """Configure console + file logging."""
    os.makedirs(config.LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(config.SYSTEM_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    return logging.getLogger(__name__)


# ── Signal Functions ─────────────────────────────

# Active signal function (set by select_signal_function)
_signal_fn = None


def select_signal_function(model_name):
    """Select which signal function to use."""
    global _signal_fn
    if model_name == "rl_ppo":
        from rl_model.live_agent import RLAgent
        agent = RLAgent()
        _signal_fn = agent.get_signal
        logger.info("Signal model: PPO Reinforcement Learning")
    else:
        _signal_fn = get_signal_ma_crossover
        logger.info("Signal model: MA Crossover")


def get_active_signal(state):
    """Get signal from the currently active signal function."""
    if _signal_fn is None:
        return get_signal_ma_crossover(state)
    return _signal_fn(state)


def get_signal_ma_crossover(state):
    """
    MA Crossover signal (baseline strategy).

    Args:
        state: dict with close, ma5, ma10, ma20, ma100, has_position, position_type, position_profit

    Returns:
        str: "BUY", "SELL", "CLOSE", or "HOLD"
    """
    close = state["close"]
    ma5 = state.get("ma5")
    ma10 = state.get("ma10")
    ma20 = state.get("ma20")
    ma100 = state.get("ma100")
    has_position = state.get("has_position", False)
    position_type = state.get("position_type")

    # Need all MAs to make a decision
    if any(v is None or (isinstance(v, float) and v != v) for v in [ma5, ma10, ma20, ma100]):
        return "HOLD"

    # ── Close signals ─────────────
    if has_position:
        if position_type == "BUY" and ma5 < ma20:
            return "CLOSE"
        if position_type == "SELL" and ma5 > ma20:
            return "CLOSE"
        return "HOLD"

    # ── Entry signals ─────────────
    # BUY: MA5 crosses above MA20 AND short-term trend is bullish (MA10 > MA100)
    if ma5 > ma20 and ma10 > ma100:
        return "BUY"

    # SELL: MA5 crosses below MA20 AND short-term trend is bearish (MA10 < MA100)
    if ma5 < ma20 and ma10 < ma100:
        return "SELL"

    return "HOLD"


# Backward-compatible alias (used by tests and evaluate.py)
def get_signal(state):
    """Backward-compatible wrapper — delegates to MA crossover."""
    return get_signal_ma_crossover(state)


# ── Build State Dict ─────────────────────────────

def build_state(df_row, has_position, position_type, position_profit):
    """Build the state dictionary from a DataFrame row."""
    return {
        "close": df_row["Close"],
        "ma5": df_row.get("MA5"),
        "ma10": df_row.get("MA10"),
        "ma20": df_row.get("MA20"),
        "ma100": df_row.get("MA100"),
        "has_position": has_position,
        "position_type": position_type,
        "position_profit": position_profit,
    }


# ── Live Trading Mode ───────────────────────────

def run_live(model_name="ma_crossover"):
    """Run the live trading loop connected to MT5 demo account."""
    from mt5_connection import MT5Connection
    import data_manager
    import trade_executor
    import account_monitor
    import trade_logger

    logger.info("=" * 60)
    logger.info(f"  MT5 TRADE EXECUTION SYSTEM — LIVE MODE ({model_name.upper()})")
    logger.info("=" * 60)

    # Select signal model
    select_signal_function(model_name)

    # Connect to MT5
    conn = MT5Connection()
    conn.connect()
    logger.info(conn.get_account_summary())

    # Verify symbol
    sym_info = data_manager.get_symbol_info()
    if sym_info is None:
        logger.error(f"Symbol {config.SYMBOL} not available. Check config.SYMBOL.")
        conn.disconnect()
        return
    logger.info(f"Trading: {sym_info['name']} | Point: {sym_info['point']} | Min Lot: {sym_info['volume_min']}")

    # Fetch initial data
    logger.info(f"Fetching last {config.INITIAL_BARS} M1 bars...")
    df = data_manager.fetch_latest_bars(count=config.INITIAL_BARS)
    if df.empty:
        logger.error("Failed to fetch initial data. Check MT5 connection.")
        conn.disconnect()
        return

    df = data_manager.compute_moving_averages(df)
    logger.info(f"Data ready: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    logger.info(f"Latest MAs — MA5: {df['MA5'].iloc[-1]:.2f}, MA20: {df['MA20'].iloc[-1]:.2f}, MA100: {df['MA100'].iloc[-1]:.2f}")

    # Log initial account state
    acct = account_monitor.get_account_info()
    if acct:
        trade_logger.log_account_snapshot(acct)

    # Graceful shutdown handler
    running = True

    def handle_shutdown(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received. Closing positions and exiting...")
        running = False

    signal.signal(signal.SIGINT, handle_shutdown)

    last_bar_time = df.index[-1] if not df.empty else None
    iteration = 0

    # ── Main Loop ────────────────────────────────
    logger.info(f"\nStarting main loop (polling every {config.LOOP_INTERVAL_SECONDS}s)...")
    logger.info("Press Ctrl+C to stop.\n")

    while running:
        try:
            iteration += 1

            # Check connection health
            if not conn.is_connected():
                logger.warning("MT5 connection lost. Reconnecting...")
                conn.reconnect()

            # Fetch latest bars and update MAs
            new_bars = data_manager.fetch_latest_bars(count=config.INITIAL_BARS)
            if not new_bars.empty:
                df = data_manager.compute_moving_averages(new_bars)
                current_bar_time = df.index[-1]
            else:
                logger.warning("Failed to fetch new bars")
                time.sleep(config.LOOP_INTERVAL_SECONDS)
                continue

            # Skip if no new bar
            if last_bar_time and current_bar_time <= last_bar_time:
                time.sleep(5)
                continue
            last_bar_time = current_bar_time

            # Get current state
            latest = df.iloc[-1]
            has_pos = account_monitor.has_open_position()
            pos_type = account_monitor.get_position_type()
            pos_profit = account_monitor.get_position_profit()

            state = build_state(latest, has_pos, pos_type, pos_profit)

            # Get signal
            action = get_active_signal(state)

            tick = data_manager.get_current_tick()
            tick_str = f"Bid: {tick['bid']:.2f} | Ask: {tick['ask']:.2f} | Spread: {tick['spread']:.0f}pts" if tick else "N/A"

            logger.info(
                f"[#{iteration}] {current_bar_time.strftime('%H:%M')} | "
                f"Close: {latest['Close']:.2f} | MA5: {latest['MA5']:.2f} | "
                f"MA20: {latest['MA20']:.2f} | {tick_str} | "
                f"Pos: {pos_type or 'NONE'} ({pos_profit:+.2f}) | "
                f"Signal: {action}"
            )

            # Execute action
            result = None

            if action == "BUY":
                ma_dict = data_manager.get_ma_dict(latest)
                sl, tp = trade_executor.calculate_sl_tp("BUY", tick["ask"] if tick else latest["Close"], ma_dict)
                result = trade_executor.open_buy(sl=sl, tp=tp)

            elif action == "SELL":
                ma_dict = data_manager.get_ma_dict(latest)
                sl, tp = trade_executor.calculate_sl_tp("SELL", tick["bid"] if tick else latest["Close"], ma_dict)
                result = trade_executor.open_sell(sl=sl, tp=tp)

            elif action == "CLOSE":
                positions = account_monitor.get_open_positions(symbol=config.SYMBOL)
                if not positions.empty:
                    for _, pos in positions.iterrows():
                        close_result = trade_executor.close_position(int(pos["ticket"]))
                        if close_result.success:
                            trade_data = close_result.to_dict()
                            trade_data["profit"] = pos["profit"]
                            trade_logger.log_trade(trade_data)

            # Log successful open trades
            if result and result.success and action in ("BUY", "SELL"):
                trade_logger.log_trade(result.to_dict())

            # Periodic account snapshot (every 10 iterations)
            if iteration % 10 == 0:
                acct = account_monitor.get_account_info()
                if acct:
                    trade_logger.log_account_snapshot(acct)
                    logger.info(f"Account: Balance=${acct['balance']:.2f}, Equity=${acct['equity']:.2f}")

            # Wait for next bar
            time.sleep(config.LOOP_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            running = False
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(config.LOOP_INTERVAL_SECONDS)

    # ── Shutdown ─────────────────────────────────
    logger.info("\nShutting down...")

    # Close all positions
    results = trade_executor.close_all_positions()
    for r in results:
        if r.success:
            trade_logger.log_trade(r.to_dict())

    # Final summary
    trade_logger.print_summary()

    # Final account snapshot
    acct = account_monitor.get_account_info()
    if acct:
        trade_logger.log_account_snapshot(acct)
        logger.info(f"Final Balance: ${acct['balance']:.2f}")

    conn.disconnect()
    logger.info("System shutdown complete.")


# ── Backtest Mode ────────────────────────────────

def run_backtest(model_name="ma_crossover"):
    """
    Run the strategy on historical CSV data without MT5.
    Simulates trades and logs performance metrics.
    """
    import data_manager
    import trade_logger

    logger.info("=" * 60)
    logger.info(f"  MT5 TRADE EXECUTION SYSTEM — BACKTEST MODE ({model_name.upper()})")
    logger.info("=" * 60)

    # Select signal model
    select_signal_function(model_name)

    # Load CSV data
    logger.info("Loading CSV data...")
    df = data_manager.load_all_csv_data()
    df = data_manager.compute_moving_averages(df)

    # Drop rows where MA100 is NaN (first 99 bars)
    df = df.dropna(subset=["MA100"])
    logger.info(f"Data ready: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Simulation state
    position = None  # None, "BUY", or "SELL"
    entry_price = 0.0
    trades = []
    balance = 10000.0  # Starting balance
    point = 0.01  # XAUUSD point
    lot_size = config.LOT_SIZE
    contract_size = 100  # XAUUSD standard: 1 lot = 100 oz

    logger.info(f"Starting backtest — Balance: ${balance:.2f}, Lot: {lot_size}")

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        state = build_state(row, position is not None, position, 0.0)
        action = get_active_signal(state)

        # Execute
        if action == "BUY" and position is None:
            entry_price = row["Close"]
            position = "BUY"
            trades.append({
                "timestamp": str(row.name),
                "action": "BUY",
                "price": entry_price,
                "profit": None,
            })

        elif action == "SELL" and position is None:
            entry_price = row["Close"]
            position = "SELL"
            trades.append({
                "timestamp": str(row.name),
                "action": "SELL",
                "price": entry_price,
                "profit": None,
            })

        elif action == "CLOSE" and position is not None:
            exit_price = row["Close"]

            if position == "BUY":
                profit = (exit_price - entry_price) * lot_size * contract_size
            else:
                profit = (entry_price - exit_price) * lot_size * contract_size

            balance += profit
            trades.append({
                "timestamp": str(row.name),
                "action": "CLOSE",
                "price": exit_price,
                "profit": round(profit, 2),
            })

            logger.debug(
                f"CLOSE {position} @ {exit_price:.2f} | "
                f"P/L: ${profit:+.2f} | Balance: ${balance:.2f}"
            )
            position = None

    # Final summary
    trades_df = pd.DataFrame(trades)
    logger.info(f"\nBacktest complete: {len(trades)} events, Final Balance: ${balance:.2f}")

    if not trades_df.empty:
        trade_logger.print_summary(trades_df)

        # Save backtest results
        os.makedirs(config.LOG_DIR, exist_ok=True)
        output_file = os.path.join(config.LOG_DIR, "backtest_results.csv")
        trades_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")


# ── Entry Point ──────────────────────────────────

if __name__ == "__main__":
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="MT5 Trade Execution System")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on CSV data (no MT5 required)")
    parser.add_argument("--model", default="ma_crossover", choices=["ma_crossover", "rl_ppo"],
                        help="Signal model: ma_crossover (default) or rl_ppo")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(model_name=args.model)
    else:
        run_live(model_name=args.model)
