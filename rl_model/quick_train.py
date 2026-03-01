"""
Quick Train — Single Month
============================
Trains PPO on one month of data with verbose output.
Shows exactly what the agent is doing: actions, trades, rewards.

Usage:
    python rl_model/quick_train.py                  # First month, 1 iteration
    python rl_model/quick_train.py --iterations 5   # 5 iterations
    python rl_model/quick_train.py --month 3        # Use 3rd month
    python rl_model/quick_train.py --ent-coef 0.1   # More exploration
"""

import os
import sys
import time
import argparse
import logging

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rl_model.trading_env import TradingEnv
import data_manager

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
TICK_EXPORT_DIR = os.path.join(
    os.environ.get("APPDATA", ""),
    "MetaQuotes", "Terminal", "Common", "Files", "tick_export"
)

logger = logging.getLogger(__name__)


class VerboseCallback(BaseCallback):
    """Logs action distribution during training."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_steps = 0
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        self.total_steps += 1
        # Track actions
        actions = self.locals.get("actions", [])
        for a in actions:
            self.action_counts[int(a)] = self.action_counts.get(int(a), 0) + 1

        # Print every 5000 steps
        if self.total_steps % 5000 == 0:
            elapsed = time.time() - self.start_time
            total = sum(self.action_counts.values())
            pcts = {k: v/total*100 for k, v in self.action_counts.items()} if total else {}
            print(f"  Step {self.total_steps:>6,} | "
                  f"HOLD:{pcts.get(0,0):5.1f}% "
                  f"BUY:{pcts.get(1,0):5.1f}% "
                  f"SELL:{pcts.get(2,0):5.1f}% "
                  f"CLOSE:{pcts.get(3,0):5.1f}% | "
                  f"Speed: {self.total_steps/elapsed:,.0f} steps/s")
        return True


def evaluate_verbose(model, env, max_steps=None):
    """Run one full episode with detailed logging."""
    obs, info = env.reset()
    total_reward = 0
    step = 0
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    if max_steps is None:
        max_steps = env.n_steps

    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        action_counts[int(action)] += 1
        step += 1

        # Log trades
        if int(action) in (1, 2):
            price = env.close_prices[min(env.current_step, env.n_steps - 1)]
            print(f"    Step {step:>6}: {action_names[int(action)]:>5} @ ${price:.2f}")
        elif int(action) == 3 and env.trade_log:
            last = env.trade_log[-1]
            print(f"    Step {step:>6}: CLOSE {last['type']} "
                  f"entry=${last['entry']:.2f} exit=${last['exit']:.2f} "
                  f"P/L=${last['pnl']:+.2f} (bal=${last['balance']:.2f})")

        if terminated or truncated:
            break

    info = env._get_info()
    total = sum(action_counts.values())

    print(f"\n  ── Evaluation Results ──")
    print(f"  Steps:        {step:,}")
    print(f"  Trades:       {info['total_trades']}")
    print(f"  Win Rate:     {info['win_rate']:.1f}%")
    print(f"  Net P/L:      ${info['net_profit']:+.2f}")
    print(f"  Balance:      ${info['balance']:.2f}")
    print(f"  Max Drawdown: {info['drawdown']*100:.2f}%")
    print(f"  Total Reward: {total_reward:+.4f}")
    print(f"  Actions:      HOLD={action_counts[0]} "
          f"BUY={action_counts[1]} SELL={action_counts[2]} CLOSE={action_counts[3]}")
    print(f"  Action %:     HOLD={action_counts[0]/total*100:.1f}% "
          f"BUY={action_counts[1]/total*100:.1f}% "
          f"SELL={action_counts[2]/total*100:.1f}% "
          f"CLOSE={action_counts[3]/total*100:.1f}%")

    return info


def main():
    parser = argparse.ArgumentParser(description="Quick single-month RL training")
    parser.add_argument("--iterations", type=int, default=1, help="Training iterations")
    parser.add_argument("--month", type=int, default=0, help="Month index (0=first)")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps per iteration")
    parser.add_argument("--ent-coef", type=float, default=0.05, help="Entropy coefficient")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Device ──
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Device: CPU")

    # ── Load data ──
    m1_file = os.path.join(TICK_EXPORT_DIR, "XAUUSD_M1_export.csv")
    if os.path.exists(m1_file):
        print(f"  Loading MT5 M1 export...")
        df = pd.read_csv(m1_file, index_col=0, parse_dates=True)
        for col in ["MA5", "MA10", "MA20", "MA100"]:
            if col in df.columns:
                df = df.drop(columns=[col])
    else:
        print(f"  Loading CSV data files...")
        df = data_manager.load_all_csv_data()

    df = data_manager.compute_moving_averages(df)
    df = df.dropna(subset=["MA100"])

    # ── Split into months ──
    chunks = [g for _, g in df.groupby(pd.Grouper(freq="ME")) if len(g) > 200]
    month_idx = min(args.month, len(chunks) - 1)
    month_data = chunks[month_idx]

    print(f"  Data: {month_data.index[0].strftime('%Y-%m')} "
          f"({len(month_data):,} bars)")

    # ── Create env ──
    train_env = TradingEnv(month_data, spread_points=25)
    eval_env = TradingEnv(month_data, spread_points=25)

    # ── Create model ──
    print(f"  Entropy: {args.ent_coef} (higher = more exploration)")
    print(f"  Steps per iteration: {args.steps:,}")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        ent_coef=args.ent_coef,
        clip_range=0.2,
        verbose=0,
        device=device,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.ReLU,
        ),
    )

    # ── Pre-training evaluation ──
    print(f"\n{'='*60}")
    print(f"  PRE-TRAINING (random policy)")
    print(f"{'='*60}")
    evaluate_verbose(model, eval_env)

    # ── Training ──
    for i in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"  TRAINING — Iteration {i+1}/{args.iterations}")
        print(f"{'='*60}")

        callback = VerboseCallback()
        t0 = time.time()

        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            reset_num_timesteps=(i == 0),
        )

        train_time = time.time() - t0
        print(f"\n  Training time: {train_time:.1f}s")

        # ── Post-training evaluation ──
        print(f"\n  ── POST-TRAINING Evaluation ──")
        evaluate_verbose(model, eval_env)

    # ── Save ──
    save_path = os.path.join(MODEL_DIR, "ppo_xauusd_quick.zip")
    model.save(save_path)
    print(f"\n  Model saved: {save_path}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
