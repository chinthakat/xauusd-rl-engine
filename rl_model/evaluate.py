"""
Model Evaluation
=================
Evaluates the trained PPO agent and compares against the MA crossover baseline.

Usage:
    python -m rl_model.evaluate                     # Evaluate best model
    python -m rl_model.evaluate --model final       # Evaluate final model
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_manager import load_all_csv_data, compute_moving_averages
from rl_model.trading_env import TradingEnv

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def evaluate_model(model_name="best", episodes=1):
    """
    Evaluate the trained PPO model on histori data.

    Args:
        model_name: "best" or "final"
        episodes:   Number of evaluation episodes (>1 for stochastic eval)

    Returns:
        dict with evaluation metrics
    """
    # Load model
    model_path = os.path.join(MODEL_DIR, f"ppo_xauusd_{model_name}")
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found: {model_path}.zip")
        print("Run training first: python -m rl_model.train")
        return None

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # Load data
    df = load_all_csv_data()
    df = compute_moving_averages(df)
    df = df.dropna(subset=["MA100"])

    # Create environment
    env = TradingEnv(df, initial_balance=10000.0, lot_size=config.LOT_SIZE)

    # Run evaluation episodes
    all_results = []
    all_trades = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        result = info.copy()
        result["episode"] = ep + 1
        result["total_reward"] = round(total_reward, 4)
        all_results.append(result)
        all_trades.extend(env.trade_log)

    # Aggregate results
    results_df = pd.DataFrame(all_results)
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    # Compute additional metrics
    metrics = _compute_metrics(results_df, trades_df, env)

    return metrics


def evaluate_baseline(df=None):
    """
    Evaluate the MA crossover baseline for comparison.
    """
    if df is None:
        df = load_all_csv_data()
        df = compute_moving_averages(df)
        df = df.dropna(subset=["MA100"])

    # Import the baseline signal function
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import get_signal, build_state

    balance = 10000.0
    position = None
    entry_price = 0.0
    lot_size = config.LOT_SIZE
    contract_size = 100
    trades = []
    spread = 0.25  # 25 points

    for i in range(1, len(df)):
        row = df.iloc[i]
        state = build_state(row, position is not None, position, 0.0)
        action = get_signal(state)

        if action == "BUY" and position is None:
            entry_price = row["Close"] + spread
            position = "BUY"
        elif action == "SELL" and position is None:
            entry_price = row["Close"]
            position = "SELL"
        elif action == "CLOSE" and position is not None:
            if position == "BUY":
                pnl = (row["Close"] - entry_price) * lot_size * contract_size
            else:
                pnl = (entry_price - row["Close"] - spread) * lot_size * contract_size
            balance += pnl
            trades.append({"pnl": pnl})
            position = None

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    winners = len(trades_df[trades_df["pnl"] > 0]) if not trades_df.empty else 0
    losers = len(trades_df[trades_df["pnl"] < 0]) if not trades_df.empty else 0
    total = len(trades_df)
    net_profit = balance - 10000.0
    gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum() if not trades_df.empty else 0
    gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum()) if not trades_df.empty else 0

    return {
        "model": "MA Crossover",
        "net_profit": round(net_profit, 2),
        "total_trades": total,
        "win_rate": round(winners / total * 100, 2) if total > 0 else 0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf"),
        "max_drawdown": _compute_drawdown(trades_df) if not trades_df.empty else 0,
        "sharpe_ratio": _compute_sharpe(trades_df) if not trades_df.empty else 0,
        "final_balance": round(balance, 2),
    }


def _compute_metrics(results_df, trades_df, env):
    """Compute comprehensive evaluation metrics."""
    if trades_df.empty:
        return {
            "model": "PPO",
            "net_profit": 0, "total_trades": 0, "win_rate": 0,
            "profit_factor": 0, "max_drawdown": 0, "sharpe_ratio": 0,
            "final_balance": 10000.0,
        }

    pnls = trades_df["pnl"].values
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]

    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0

    return {
        "model": "PPO",
        "net_profit": round(results_df["net_profit"].mean(), 2),
        "total_trades": int(results_df["total_trades"].mean()),
        "win_rate": round(len(winners) / len(pnls) * 100, 2) if len(pnls) > 0 else 0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf"),
        "max_drawdown": _compute_drawdown(trades_df),
        "sharpe_ratio": _compute_sharpe(trades_df),
        "final_balance": round(results_df["balance"].iloc[-1], 2),
    }


def _compute_drawdown(trades_df):
    """Max drawdown from trade P/L series."""
    if trades_df.empty or "pnl" not in trades_df.columns:
        return 0.0
    cum = trades_df["pnl"].cumsum()
    peak = cum.cummax()
    dd = (peak - cum).max()
    return round(dd, 2)


def _compute_sharpe(trades_df, risk_free=0):
    """Annualized Sharpe ratio from trade P/Ls."""
    if trades_df.empty or "pnl" not in trades_df.columns:
        return 0.0
    pnls = trades_df["pnl"]
    if pnls.std() == 0:
        return 0.0
    return round((pnls.mean() / pnls.std()) * np.sqrt(252), 4)


def print_comparison(ppo_metrics, baseline_metrics):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("            MODEL COMPARISON: PPO vs MA Crossover")
    print("=" * 70)
    print(f"{'Metric':<20} {'PPO':>20} {'MA Crossover':>20}")
    print("-" * 70)

    for key in ["net_profit", "total_trades", "win_rate", "profit_factor",
                "max_drawdown", "sharpe_ratio", "final_balance"]:
        ppo_val = ppo_metrics.get(key, "N/A")
        base_val = baseline_metrics.get(key, "N/A")

        if isinstance(ppo_val, float):
            ppo_str = f"${ppo_val:,.2f}" if key in ("net_profit", "max_drawdown", "final_balance") else f"{ppo_val}"
            base_str = f"${base_val:,.2f}" if key in ("net_profit", "max_drawdown", "final_balance") else f"{base_val}"
        else:
            ppo_str = str(ppo_val)
            base_str = str(base_val)

        # Highlight winner
        better = ""
        if isinstance(ppo_val, (int, float)) and isinstance(base_val, (int, float)):
            if key == "max_drawdown":
                better = " ★" if ppo_val < base_val else ""
            elif key != "total_trades":
                better = " ★" if ppo_val > base_val else ""

        print(f"{key:<20} {ppo_str:>20} {base_str:>20}{better}")

    print("=" * 70)


# ── Entry point ──────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate PPO trading agent")
    parser.add_argument("--model", default="best", choices=["best", "final"], help="Which model to evaluate")
    parser.add_argument("--episodes", type=int, default=1, help="Number of eval episodes")
    args = parser.parse_args()

    # Evaluate PPO
    print("Evaluating PPO model...")
    ppo_metrics = evaluate_model(model_name=args.model, episodes=args.episodes)

    if ppo_metrics:
        # Evaluate baseline
        print("\nEvaluating MA Crossover baseline...")
        baseline_metrics = evaluate_baseline()

        # Compare
        print_comparison(ppo_metrics, baseline_metrics)

        # Save results
        results_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "evaluation_results.csv"
        )
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        pd.DataFrame([ppo_metrics, baseline_metrics]).to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
