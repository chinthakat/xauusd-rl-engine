"""
Parallel GPU Training
======================
Trains PPO across monthly data chunks in parallel using SubprocVecEnv.
Each month runs in its own subprocess, PPO model runs on GPU.

Usage:
    python -m rl_model.parallel_train                     # 10 iterations
    python -m rl_model.parallel_train --iterations 5      # 5 iterations
    python -m rl_model.parallel_train --cpu                # Force CPU
"""

import os
import sys
import time
import logging
import argparse

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.trading_env import TradingEnv
import data_manager

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "tensorboard")
TICK_EXPORT_DIR = os.path.join(
    os.environ.get("APPDATA", ""), "MetaQuotes", "Terminal", "Common", "Files", "tick_export"
)


def split_by_month(df):
    """Split a DataFrame with DatetimeIndex into monthly chunks."""
    groups = df.groupby(pd.Grouper(freq="ME"))
    chunks = []
    for name, group in groups:
        if len(group) > 200:  # Skip months with too few bars
            chunks.append(group)
    return chunks


def make_env_fn(df_chunk, spread_points=25):
    """Create a factory function for SubprocVecEnv."""
    def _init():
        env = TradingEnv(
            df=df_chunk,
            initial_balance=10000.0,
            lot_size=0.01,
            contract_size=100,
            spread_points=spread_points,
        )
        return env
    return _init


class ParallelTrainingCallback(BaseCallback):
    """Track training progress across iterations."""

    def __init__(self, total_iterations, verbose=1):
        super().__init__(verbose)
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.iteration_start = None
        self.step_count = 0

    def set_iteration(self, i):
        self.current_iteration = i
        self.iteration_start = time.time()
        self.step_count = 0

    def _on_step(self):
        self.step_count += 1
        if self.step_count % 10000 == 0 and self.verbose:
            elapsed = time.time() - self.iteration_start
            sps = self.step_count / max(elapsed, 0.1)
            print(f"\r  Iteration {self.current_iteration+1}/{self.total_iterations} | "
                  f"Steps: {self.step_count:>8,} | "
                  f"Speed: {sps:,.0f} steps/sec | "
                  f"Time: {elapsed:.0f}s", end="", flush=True)
        return True


def parallel_train(
    iterations=10,
    force_cpu=False,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    spread_points=25,
):
    """
    Train PPO model in parallel across monthly data chunks.

    Args:
        iterations:     Number of full passes over all monthly data
        force_cpu:      Force CPU even if GPU available
        n_steps:        Rollout steps per env before update
        batch_size:     Minibatch size for PPO updates
        spread_points:  Spread to use in simulation
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── Device selection ──
    if torch.cuda.is_available() and not force_cpu:
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("  Running on CPU")

    # ── Load data ──
    m1_file = os.path.join(TICK_EXPORT_DIR, "XAUUSD_M1_export.csv")
    if os.path.exists(m1_file):
        print(f"  Loading MT5 M1 export...")
        df = pd.read_csv(m1_file, index_col=0, parse_dates=True)
        # Drop any pre-filled MAs and recompute with Python
        for col in ["MA5", "MA10", "MA20", "MA100"]:
            if col in df.columns:
                df = df.drop(columns=[col])
    else:
        print(f"  Loading CSV data files...")
        df = data_manager.load_all_csv_data()

    df = data_manager.compute_moving_averages(df)
    df = df.dropna(subset=["MA100"])
    print(f"  Total data: {len(df):,} M1 bars ({df.index[0].date()} to {df.index[-1].date()})")

    # ── Split into monthly chunks ──
    monthly_chunks = split_by_month(df)
    n_months = len(monthly_chunks)
    print(f"  Monthly chunks: {n_months}")
    for i, chunk in enumerate(monthly_chunks):
        print(f"    [{i+1:>2}] {chunk.index[0].strftime('%Y-%m')}: {len(chunk):>6,} bars")

    # Limit parallel envs based on CPU cores (leave 2 for system)
    import multiprocessing
    max_envs = min(n_months, multiprocessing.cpu_count() - 2, 14)
    env_chunks = monthly_chunks[:max_envs]
    print(f"\n  Parallel environments: {len(env_chunks)} (max {max_envs})")

    # ── Create vectorized environment ──
    print(f"  Creating SubprocVecEnv...")
    env_fns = [make_env_fn(chunk, spread_points) for chunk in env_chunks]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # ── Create or load PPO model ──
    best_model_path = os.path.join(MODEL_DIR, "ppo_xauusd_best.zip")
    if os.path.exists(best_model_path):
        print(f"  Loading existing model: {best_model_path}")
        model = PPO.load(best_model_path, env=vec_env, device=device)
        model.learning_rate = learning_rate
    else:
        print(f"  Creating new PPO model on {device}")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            verbose=0,
            device=device,
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 128], vf=[256, 128]),
                activation_fn=torch.nn.ReLU,
            ),
        )

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"  PARALLEL GPU TRAINING")
    print(f"  {iterations} iterations x {len(env_chunks)} envs x {n_steps} steps")
    print(f"  Total steps per iteration: {len(env_chunks) * n_steps:,}")
    print(f"{'='*60}\n")

    callback = ParallelTrainingCallback(iterations)
    best_reward = -float("inf")
    start_time = time.time()

    for i in range(iterations):
        callback.set_iteration(i)
        iter_start = time.time()

        # Steps per iteration = n_steps * num_envs (one full rollout per env)
        total_timesteps = n_steps * len(env_chunks)

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name=f"parallel_ppo",
            progress_bar=False,
        )

        iter_time = time.time() - iter_start

        # Evaluate: run one episode per env and average reward
        rewards = []
        obs = vec_env.reset()
        done_flags = [False] * len(env_chunks)
        episode_rewards = np.zeros(len(env_chunks))

        for _ in range(max(len(c) for c in env_chunks)):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rews, dones, infos = vec_env.step(actions)
            episode_rewards += rews
            for j, done in enumerate(dones):
                if done and not done_flags[j]:
                    rewards.append(episode_rewards[j])
                    done_flags[j] = True
            if all(done_flags):
                break

        mean_reward = np.mean(rewards) if rewards else 0
        print(f"\n  Iter {i+1}/{iterations}: "
              f"Mean reward: {mean_reward:+.2f} | "
              f"Time: {iter_time:.1f}s | "
              f"Speed: {total_timesteps/iter_time:,.0f} steps/sec")

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            model.save(best_model_path)
            print(f"  >>> New best model saved (reward: {mean_reward:+.2f})")

        # Save checkpoint every iteration
        checkpoint_path = os.path.join(MODEL_DIR, f"ppo_xauusd_iter{i+1}.zip")
        model.save(checkpoint_path)

    # ── Final save ──
    final_path = os.path.join(MODEL_DIR, "ppo_xauusd_final.zip")
    model.save(final_path)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best reward: {best_reward:+.2f}")
    print(f"  Models saved to: {MODEL_DIR}")
    print(f"{'='*60}")

    vec_env.close()
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Parallel GPU Training")
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per env")
    parser.add_argument("--spread", type=int, default=25, help="Spread in points")
    args = parser.parse_args()

    parallel_train(
        iterations=args.iterations,
        force_cpu=args.cpu,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        spread_points=args.spread,
    )
