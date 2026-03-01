"""
PPO Training Loop
==================
Trains the PPO agent on XAUUSD M1 historical data.
Uses GPU (CUDA) when available for accelerated training.

Usage:
    python -m rl_model.train                    # Train 100 episodes
    python -m rl_model.train --episodes 50      # Train 50 episodes
    python -m rl_model.train --cpu              # Force CPU
"""

import os
import sys
import time
import argparse
import logging

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_manager import load_all_csv_data, compute_moving_averages
from rl_model.trading_env import TradingEnv

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_xauusd_best")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_xauusd_final")
LOG_DIR_TB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "tensorboard")


# ── Training Callback ───────────────────────────

class TradingCallback(BaseCallback):
    """
    Custom callback for logging episode stats and saving best model.
    """

    def __init__(self, eval_env, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.best_net_profit = -float("inf")
        self.episode_count = 0
        self.episode_rewards = []
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"\n{'='*70}")
        print(f"  PPO TRAINING STARTED")
        print(f"  Device: {self.model.device}")
        print(f"  Total timesteps: {self.model._total_timesteps:,}")
        print(f"{'='*70}\n")

    def _on_step(self):
        # Check for episode completion via info
        for info in self.locals.get("infos", []):
            if "episode" in info or info.get("total_trades", 0) > 0:
                pass  # SB3 doesn't always expose episode end in infos

        # Periodic evaluation
        if self.n_calls % self.check_freq == 0:
            net_profit = self._evaluate()

            elapsed = time.time() - self.start_time
            steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0

            if self.verbose:
                print(
                    f"  Step {self.n_calls:>8,d} | "
                    f"Eval Profit: ${net_profit:>+8.2f} | "
                    f"Best: ${self.best_net_profit:>+8.2f} | "
                    f"Speed: {steps_per_sec:,.0f} steps/s"
                )

            # Save best model
            if net_profit > self.best_net_profit:
                self.best_net_profit = net_profit
                os.makedirs(MODEL_DIR, exist_ok=True)
                self.model.save(BEST_MODEL_PATH)
                if self.verbose:
                    print(f"  ★ New best model saved! (profit: ${net_profit:+.2f})")

        return True

    def _evaluate(self):
        """Run a quick evaluation episode."""
        obs, _ = self.eval_env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        return info.get("net_profit", 0)

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        print(f"\n{'='*70}")
        print(f"  TRAINING COMPLETE")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"  Best Net Profit: ${self.best_net_profit:+.2f}")
        print(f"{'='*70}\n")


# ── Main Training Function ──────────────────────

def train(
    episodes=100,
    mode="sim",
    bridge_folder="rl_bridge",
    force_cpu=False,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=None,
):
    """
    Train the PPO agent.

    Args:
        episodes:       Number of training episodes
        mode:           'sim' (CSV-simulated) or 'mt5' (MT5 Strategy Tester)
        bridge_folder:  IPC folder name for MT5 mode
        force_cpu:      If True, use CPU even if GPU is available
        (remaining args: PPO hyperparameters)
    """
    # ── Setup device ─────────────────────────
    if force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = "cpu"
        print("No GPU detected, using CPU")

    print(f"Training device: {device}")
    print(f"Training mode: {mode.upper()}")
    print(f"PyTorch version: {torch.__version__}")

    # ── Create environments based on mode ────
    if mode == "mt5":
        from rl_model.mt5_backtest_env import MT5BacktestEnv

        print(f"\nMT5 Bridge mode — folder: {bridge_folder}")
        print("Waiting for MT5 Strategy Tester to start RLBridge EA...")

        # For MT5 mode: single env, no DummyVecEnv wrapping for eval
        eval_env = MT5BacktestEnv(bridge_folder=bridge_folder)

        if not eval_env.bridge.wait_for_ea_ready(timeout=60):
            print("ERROR: EA not ready. Start the Strategy Tester with RLBridge EA first.")
            return None

        # Training env (wrapped for SB3)
        def make_mt5_env():
            return MT5BacktestEnv(bridge_folder=bridge_folder)
        train_env = DummyVecEnv([make_mt5_env])

        # For MT5, episode length is determined by Strategy Tester date range
        # Use a reasonable estimate for total_timesteps
        estimated_steps_per_ep = 2000  # ~1 month M15 or ~1 week M1
        total_timesteps = episodes * estimated_steps_per_ep
        n_steps = min(n_steps, estimated_steps_per_ep)  # Don't overshoot episode

        print(f"Estimated steps/episode: ~{estimated_steps_per_ep:,}")

    else:  # sim mode
        print("\nLoading CSV data...")
        df = load_all_csv_data()
        df = compute_moving_averages(df)
        df = df.dropna(subset=["MA100"])
        print(f"Data: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

        print("Creating environments...")
        def make_train_env():
            return TradingEnv(df, initial_balance=10000.0, lot_size=config.LOT_SIZE)
        train_env = DummyVecEnv([make_train_env])
        eval_env = TradingEnv(df, initial_balance=10000.0, lot_size=config.LOT_SIZE)

        total_timesteps = episodes * len(df)

    print(f"Episodes: {episodes}")
    print(f"Total timesteps: {total_timesteps:,}")

    # ── Policy network ───────────────────────
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
        )

    # ── Initialize PPO ───────────────────────
    print("\nInitializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
        tensorboard_log=LOG_DIR_TB,
    )

    print(f"Policy network:\n{model.policy}")

    # ── Training callback ────────────────────
    steps_per_ep = total_timesteps // max(episodes, 1)
    eval_freq = max(steps_per_ep // 2, 1000)
    callback = TradingCallback(eval_env, check_freq=eval_freq, verbose=1)

    # ── Train ────────────────────────────────
    print(f"\nStarting training ({episodes} episodes on {device}, mode={mode})...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # ── Save final model ─────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to: {FINAL_MODEL_PATH}")
    print(f"Best model saved to:  {BEST_MODEL_PATH}")

    # Cleanup
    train_env.close()

    return model


# ── Entry point ──────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Train PPO trading agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--mode", default="sim", choices=["sim", "mt5"],
                        help="Training mode: sim (CSV-simulated) or mt5 (MT5 Strategy Tester)")
    parser.add_argument("--bridge-folder", default="rl_bridge",
                        help="IPC folder name in Common\\Files (for MT5 mode)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    model = train(
        episodes=args.episodes,
        mode=args.mode,
        bridge_folder=args.bridge_folder,
        force_cpu=args.cpu,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
