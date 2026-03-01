"""
XAU RL Trainer — Interactive CLI
==================================
Interactive menu for training, evaluating, and managing RL trading models.

Usage:
    python trainer.py
"""

import os
import sys
import time
import glob
import logging

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from rl_model.trading_env import TradingEnv
import data_manager

logging.basicConfig(level=logging.WARNING)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TICK_EXPORT_DIR = os.path.join(
    os.environ.get("APPDATA", ""),
    "MetaQuotes", "Terminal", "Common", "Files", "tick_export"
)

# ═══════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════

_cached_df = None
_cached_chunks = None


def load_data():
    """Load and cache the full M1 dataset."""
    global _cached_df, _cached_chunks

    if _cached_df is not None:
        return _cached_df, _cached_chunks

    m1_file = os.path.join(TICK_EXPORT_DIR, "XAUUSD_M1_export.csv")
    if os.path.exists(m1_file):
        print("  Loading MT5 M1 export...")
        df = pd.read_csv(m1_file, index_col=0, parse_dates=True)
        for col in ["MA5", "MA10", "MA20", "MA100"]:
            if col in df.columns:
                df = df.drop(columns=[col])
    else:
        print("  Loading CSV data files...")
        df = data_manager.load_all_csv_data()

    df = data_manager.compute_moving_averages(df)
    df = df.dropna(subset=["MA100"])

    # Split into months
    chunks = []
    for name, group in df.groupby(pd.Grouper(freq="ME")):
        if len(group) > 200:
            chunks.append(group)

    _cached_df = df
    _cached_chunks = chunks
    return df, chunks


def split_train_eval(chunks, eval_months=2):
    """Split monthly chunks into train/eval sets. Last N months = eval."""
    if len(chunks) <= eval_months:
        return chunks, []
    train = chunks[:-eval_months]
    eval_ = chunks[-eval_months:]
    return train, eval_


# ═══════════════════════════════════════════════════════════
#  Model Management
# ═══════════════════════════════════════════════════════════

def list_models():
    """List all saved models."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = sorted(glob.glob(os.path.join(MODEL_DIR, "*.zip")))

    if not models:
        print("\n  No saved models found.\n")
        return []

    print(f"\n  {'#':<4} {'Model Name':<35} {'Size':>8}  {'Modified'}")
    print(f"  {'─'*4} {'─'*35} {'─'*8}  {'─'*20}")

    for i, path in enumerate(models):
        name = os.path.basename(path).replace(".zip", "")
        size_kb = os.path.getsize(path) / 1024
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        print(f"  {i+1:<4} {name:<35} {size_kb:>6.0f}KB  {mtime}")

    print()
    return models


def select_or_create_model(env):
    """Interactive: pick an existing model or create a new one."""
    models = list_models()

    print("  [N] Create new model")
    if models:
        print("  [1-{}] Load existing model".format(len(models)))
    print()

    choice = input("  Choice: ").strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if choice.upper() == "N" or not models:
        # New model
        name = input("  Model name (default: ppo_xauusd): ").strip()
        if not name:
            name = "ppo_xauusd"
        name = name.replace(" ", "_").replace(".zip", "")

        ent = input("  Entropy coefficient (default 0.01): ").strip()
        ent_coef = float(ent) if ent else 0.01

        print(f"\n  Creating new model '{name}' on {device} (entropy={ent_coef})")

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            ent_coef=ent_coef,
            clip_range=0.2,
            verbose=0,
            device=device,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 128], vf=[256, 128]),
                activation_fn=torch.nn.ReLU,
            ),
        )
        return model, name

    else:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            path = models[idx]
            name = os.path.basename(path).replace(".zip", "")
            print(f"\n  Loading '{name}' on {device}...")
            model = PPO.load(path, env=env, device=device)
            return model, name
        else:
            print("  Invalid choice.")
            return None, None


# ═══════════════════════════════════════════════════════════
#  Training Callback
# ═══════════════════════════════════════════════════════════

class TrainCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.steps = 0
        self.t0 = None

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self):
        self.steps += 1
        for a in self.locals.get("actions", []):
            self.action_counts[int(a)] = self.action_counts.get(int(a), 0) + 1

        if self.steps % 5000 == 0:
            elapsed = time.time() - self.t0
            total = sum(self.action_counts.values()) or 1
            pcts = {k: v / total * 100 for k, v in self.action_counts.items()}
            print(f"    Step {self.steps:>6,} | "
                  f"H:{pcts[0]:4.1f}% B:{pcts[1]:4.1f}% "
                  f"S:{pcts[2]:4.1f}% C:{pcts[3]:4.1f}% | "
                  f"{self.steps / elapsed:,.0f} stp/s")
        return True


# ═══════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════

def evaluate_model(model, env, label="", verbose=True):
    """Run one full episode and return results."""
    obs, _ = env.reset()
    total_reward = 0
    actions = {0: 0, 1: 0, 2: 0, 3: 0}

    for _ in range(env.n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
        total_reward += reward
        actions[int(action)] += 1
        if done:
            break

    info = env._get_info()
    total = sum(actions.values()) or 1

    if verbose:
        print(f"\n  ── {label} ──")
        print(f"  Trades:       {info['total_trades']:>6}")
        print(f"  Win Rate:     {info['win_rate']:>6.1f}%")
        print(f"  Net P/L:      ${info['net_profit']:>+9.2f}")
        print(f"  Balance:      ${info['balance']:>9.2f}")
        print(f"  Drawdown:     {info['drawdown']*100:>6.2f}%")
        print(f"  P/F Ratio:    {info['profit_factor']:>6.2f}")
        print(f"  Reward:       {total_reward:>+9.4f}")
        print(f"  Actions:      H={actions[0]/total*100:.0f}% "
              f"B={actions[1]/total*100:.0f}% "
              f"S={actions[2]/total*100:.0f}% "
              f"C={actions[3]/total*100:.0f}%")

    return info, total_reward


# ═══════════════════════════════════════════════════════════
#  Menu Actions
# ═══════════════════════════════════════════════════════════

def menu_list_models():
    """Option 1: List models."""
    list_models()


def menu_train():
    """Option 2: Train a model."""
    df, chunks = load_data()
    train_chunks, eval_chunks_hold = split_train_eval(chunks, eval_months=2)

    # Show available months with train/eval labels
    print(f"\n  Available months ({len(chunks)} total, {len(train_chunks)} train / {len(eval_chunks_hold)} eval-holdout):")
    for i, c in enumerate(chunks):
        tag = "EVAL" if i >= len(train_chunks) else "TRAIN"
        print(f"    [{i:>2}] {c.index[0].strftime('%Y-%m')}  ({len(c):>6,} bars) [{tag}]")

    # Training mode
    print(f"\n  Training mode:")
    print(f"  [S] Single month")
    print(f"  [C] Combined (multi-month, single env)")
    print(f"  [M] Multi-month (parallel subprocesses)")
    mode = input("  Choice: ").strip().upper()

    if mode == "S":
        month_idx = int(input(f"  Month index [0-{len(train_chunks)-1}]: ").strip())
        month_idx = max(0, min(month_idx, len(train_chunks) - 1))
        selected = [train_chunks[month_idx]]
        print(f"  Selected: {selected[0].index[0].strftime('%Y-%m')}")

    elif mode in ("M", "C"):
        range_input = input(f"  Month range [e.g. 0-11 or 0 5] (0-{len(train_chunks)-1}): ").strip()
        if "-" in range_input:
            parts = range_input.split("-")
            start, end = int(parts[0]), int(parts[1])
        elif " " in range_input:
            parts = range_input.split()
            start, end = int(parts[0]), int(parts[1])
        else:
            start = int(range_input)
            end = int(input(f"  End month index [{start}-{len(train_chunks)-1}]: ").strip())
        start = max(0, min(start, len(train_chunks) - 1))
        end = max(start, min(end, len(train_chunks) - 1))
        selected = train_chunks[start:end + 1]
        print(f"  Selected: {selected[0].index[0].strftime('%Y-%m')} to "
              f"{selected[-1].index[-1].strftime('%Y-%m')} ({len(selected)} months)")
    else:
        print("  Invalid choice.")
        return

    iterations = int(input("  Iterations (default 5): ").strip() or "5")
    steps_per = int(input("  Steps per iteration (default 100000): ").strip() or "100000")

    # Create env(s)
    if mode == "S":
        env = TradingEnv(selected[0], spread_points=25,
                         random_start=True, random_spread=True, random_slippage=True,
                         episode_length=5000)
    elif mode == "C":
        # Combined: concatenate all selected months into one big DataFrame
        combined_df = pd.concat(selected)
        total_bars = len(combined_df)
        print(f"  Combined: {total_bars:,} bars in single env (episode_length=5000)")
        env = TradingEnv(combined_df, spread_points=25,
                         random_start=True, random_spread=True, random_slippage=True,
                         episode_length=5000)
    else:  # "M" — parallel
        import multiprocessing
        max_envs = min(len(selected), multiprocessing.cpu_count() - 2)
        use = selected[:max_envs]

        def make_fn(chunk):
            def _init():
                return TradingEnv(chunk, spread_points=25,
                                 random_start=True, random_spread=True, random_slippage=True,
                                 episode_length=3000)
            return _init

        print(f"  Creating {len(use)} parallel environments...")
        env = SubprocVecEnv([make_fn(c) for c in use])
        env = VecMonitor(env)

    # Select or create model
    model, model_name = select_or_create_model(env)
    if model is None:
        return

    # Train
    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_name}")
    print(f"  {iterations} iterations x {steps_per:,} steps")
    print(f"  Randomization: start=ON spread=ON slippage=ON")
    print(f"{'='*60}")

    for i in range(iterations):
        print(f"\n  ── Iteration {i+1}/{iterations} ──")
        cb = TrainCallback()
        t0 = time.time()

        model.learn(
            total_timesteps=steps_per,
            callback=cb,
            reset_num_timesteps=(i == 0),
        )

        elapsed = time.time() - t0
        print(f"    Done ({elapsed:.1f}s)")

        # Eval on TRAINING month (no randomization)
        eval_env = TradingEnv(selected[0], spread_points=25,
                              random_start=False, random_spread=False, random_slippage=False)
        evaluate_model(model, eval_env,
                       label=f"Iter {i+1} [TRAIN] {selected[0].index[0].strftime('%Y-%m')}")

        # Eval on HOLDOUT month (out of sample)
        if eval_chunks_hold:
            oos_env = TradingEnv(eval_chunks_hold[0], spread_points=25,
                                random_start=False, random_spread=False, random_slippage=False)
            evaluate_model(model, oos_env,
                           label=f"Iter {i+1} [EVAL]  {eval_chunks_hold[0].index[0].strftime('%Y-%m')}")

    # Save
    save_path = os.path.join(MODEL_DIR, f"{model_name}.zip")
    model.save(save_path)
    print(f"\n  Model saved: {save_path}")

    # Close parallel envs
    if hasattr(env, 'close'):
        env.close()


def menu_evaluate():
    """Option 3: Evaluate a model across months."""
    df, chunks = load_data()
    train_chunks, eval_chunks_hold = split_train_eval(chunks, eval_months=2)
    models = list_models()

    if not models:
        print("  No models to evaluate.")
        return

    choice = input(f"  Select model [1-{len(models)}]: ").strip()
    idx = int(choice) - 1
    if idx < 0 or idx >= len(models):
        print("  Invalid choice.")
        return

    path = models[idx]
    name = os.path.basename(path).replace(".zip", "")

    # Which months to evaluate on?
    print(f"\n  Evaluate on:")
    print(f"  [A] All months")
    print(f"  [T] Train months only")
    print(f"  [E] Eval/holdout months only")
    print(f"  [R] Custom range")
    eval_mode = input("  Choice: ").strip().upper()

    if eval_mode == "R":
        start = int(input(f"  Start month [0-{len(chunks)-1}]: ").strip())
        end = int(input(f"  End month [{start}-{len(chunks)-1}]: ").strip())
        eval_chunks = chunks[max(0, start):min(end + 1, len(chunks))]
    elif eval_mode == "T":
        eval_chunks = train_chunks
    elif eval_mode == "E":
        eval_chunks = eval_chunks_hold
    else:
        eval_chunks = chunks

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate — NO randomization
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {name}")
    print(f"  {len(eval_chunks)} months on {device} (no randomization)")
    print(f"{'='*60}")

    results = []
    for i, chunk in enumerate(eval_chunks):
        env = TradingEnv(chunk, spread_points=25,
                         random_start=False, random_spread=False, random_slippage=False)
        model = PPO.load(path, env=env, device=device)

        month_label = chunk.index[0].strftime('%Y-%m')
        # Mark if this is in-sample or out-of-sample
        is_oos = chunk.index[0] >= eval_chunks_hold[0].index[0] if eval_chunks_hold else False
        tag = "OOS" if is_oos else "IS"

        info, reward = evaluate_model(model, env, label=month_label, verbose=False)
        info["month"] = month_label
        info["reward"] = reward
        info["sample"] = tag
        results.append(info)

        status = "+" if info["net_profit"] > 0 else "-"
        print(f"  [{i+1:>2}] {month_label} [{tag}]: "
              f"P/L=${info['net_profit']:>+8.2f} | "
              f"Trades={info['total_trades']:>4} | "
              f"WR={info['win_rate']:>5.1f}% | "
              f"DD={info['drawdown']*100:>5.2f}% {status}")

    # Summary
    total_pnl = sum(r["net_profit"] for r in results)
    total_trades = sum(r["total_trades"] for r in results)
    avg_wr = np.mean([r["win_rate"] for r in results if r["total_trades"] > 0]) if results else 0
    max_dd = max(r["drawdown"] for r in results) if results else 0
    winning_months = sum(1 for r in results if r["net_profit"] > 0)

    # Split IS vs OOS
    is_results = [r for r in results if r.get("sample") == "IS"]
    oos_results = [r for r in results if r.get("sample") == "OOS"]

    print(f"\n  {'─'*55}")
    print(f"  TOTAL P/L:       ${total_pnl:>+10.2f}")
    print(f"  Total Trades:    {total_trades:>10}")
    print(f"  Avg Win Rate:    {avg_wr:>10.1f}%")
    print(f"  Max Drawdown:    {max_dd*100:>10.2f}%")
    print(f"  Winning Months:  {winning_months}/{len(results)}")
    if is_results:
        is_pnl = sum(r["net_profit"] for r in is_results)
        print(f"  In-Sample P/L:   ${is_pnl:>+10.2f}  ({len(is_results)} months)")
    if oos_results:
        oos_pnl = sum(r["net_profit"] for r in oos_results)
        print(f"  Out-of-Sample:   ${oos_pnl:>+10.2f}  ({len(oos_results)} months)")
    print(f"  {'─'*55}")


# ═══════════════════════════════════════════════════════════
#  Main Menu
# ═══════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = "CUDA: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    while True:
        print(f"\n{'='*60}")
        print(f"  XAU RL TRAINER")
        print(f"  Device: {device}")
        print(f"{'='*60}")
        print(f"  [1] List saved models")
        print(f"  [2] Train model")
        print(f"  [3] Evaluate model")
        print(f"  [Q] Quit")
        print()

        choice = input("  > ").strip().upper()

        if choice == "1":
            menu_list_models()
        elif choice == "2":
            menu_train()
        elif choice == "3":
            menu_evaluate()
        elif choice == "Q":
            print("  Goodbye!")
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
