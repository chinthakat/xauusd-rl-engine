# RL Trading System — Class Reference

## 1. TradingEnv (Gymnasium Environment)

**File**: [trading_env.py](file:///c:/Projects/Trading/XAU_LEarningModel/rl_model/trading_env.py)

```python
class TradingEnv(gym.Env):
    """XAUUSD M1 Trading Environment"""

    # ── Action Space ──
    HOLD  = 0   # Do nothing
    BUY   = 1   # Open long / flip short→long
    SELL  = 2   # Open short / flip long→short
    CLOSE = 3   # Close current position

    # ── Constructor ──
    def __init__(self, df, initial_balance=10000.0, lot_size=0.01,
                 contract_size=100, spread_points=25, leverage=100):
        # Observation: Box(14,) float32, range [-10, 10]
        # Action:      Discrete(4)
        # min_hold_steps = 30 bars before close/flip allowed

    # ── Core Methods ──
    def reset(seed, options) → (obs, info)
    def step(action)         → (obs, reward, terminated, truncated, info)
    def render()             → prints step/price/position/P&L

    # ── Position Management (private) ──
    def _can_open(price)     → bool   # Check free_margin >= required_margin
    def _can_close()         → bool   # Check steps_in_position >= min_hold_steps
    def _open_position(dir, price)    # Set position, calculate margin
    def _close_position(price) → pnl  # Realize P/L, release margin, log trade

    # ── Observation (private) ──
    def _get_obs_fast()      → np.array(14,)  # Market features + position state
    def _get_info()          → dict           # Balance, equity, trades, win_rate, etc.
```

### State Variables

```python
# Account
self.balance        # Cash balance (changes only on trade close)
self.equity         # balance + unrealized_pnl
self.margin_used    # Locked margin for open position
self.free_margin    # equity - margin_used

# Position
self.position       # 0=flat, 1=long, -1=short
self.entry_price    # Price at which position was opened
self.unrealized_pnl # Current floating P/L
self.prev_unrealized_pnl  # Previous step's P/L (for direction tracking)

# Tracking
self.steps_in_position   # Bars held in current position
self.steps_since_trade   # Bars since last open/close (for inactivity penalty)
self.total_trades, self.winning_trades, self.losing_trades
self.total_profit, self.total_loss
self.peak_balance, self.peak_equity
```

### Position Rules

```
Action    Current Position    Result
─────────────────────────────────────────────────────────
BUY       Flat                Open LONG (if margin OK)
BUY       SHORT (≥30 bars)    Close SHORT → Open LONG
BUY       SHORT (<30 bars)    Ignored (too early)
BUY       LONG                Ignored (already long)
SELL      Flat                Open SHORT (if margin OK)
SELL      LONG (≥30 bars)     Close LONG → Open SHORT
SELL      LONG (<30 bars)     Ignored (too early)
SELL      SHORT               Ignored (already short)
CLOSE     Any (≥30 bars)      Close position
CLOSE     Any (<30 bars)      Ignored (too early)
HOLD      Any                 Do nothing
```

### Margin Calculation

```python
required_margin = (price × lot_size × contract_size) / leverage
# Example: ($2,626 × 0.01 × 100) / 100 = $26.26

# Balance only changes on close:
#   Open:  margin reserved from free_margin
#   Close: balance += realized_pnl, margin released
```

---

## 2. Reward Function & Transaction Cost Logic

All computed inside `TradingEnv.step()`. Seven components:

### Component 1 — Realized P/L (on trade close)

```python
if trade_closed and realized_pnl != 0:
    reward += log(balance_after / balance_before) × 100
```

Log return makes agent more loss-averse than gain-seeking.

### Component 2 — Profitable Trade Bonus

```python
if trade_closed and realized_pnl > 0:
    reward += 0.02  # Only winning trades get bonus
```

### Component 3 — Smart HOLD (while in position)

```python
pnl_change = unrealized_pnl - prev_unrealized_pnl

if action == HOLD:
    if pnl_change > 0:  reward += min(+0.005, pnl_change × 0.01)  # Let winners run
    if pnl_change < 0:  reward += max(-0.003, pnl_change × 0.01)  # Cut losers

reward -= 0.0002  # Base holding cost (always, any action while in position)
```

### Component 4 — Spread / Transaction Cost

```python
if flip (close + open):  reward -= spread × lot_contract × 0.02  # Double cost
elif open only:          reward -= spread × lot_contract × 0.01  # Single cost
# Example: 25pts × 0.01 × 100 × 0.01 = -0.025 per trade
```

### Component 5 — Inactivity Penalty (while flat)

```python
if flat and not_trading:
    steps_since_trade += 1
    if steps_since_trade > 200:
        penalty = min(0.003, 0.0003 × (steps_since_trade - 200) / 200)
        reward -= penalty
# Ramps: 0 at 200 steps → -0.003/step at 2200 steps
```

### Component 6 — Churning Penalty

```python
if total_trades > 0:
    avg_duration = current_step / total_trades
    if avg_duration < 50:  # Trading faster than 1 per 50 bars
        reward -= 0.002 × (50 - avg_duration) / 50
```

### Component 7 — Drawdown Penalty

```python
if equity_drawdown > 5% and reward < 0:
    reward *= 1.5  # Amplify negative reward during drawdown
```

### Summary Table

| # | Component | Trigger | Range | Purpose |
|:-:|-----------|---------|:-----:|---------|
| 1 | Realized P/L | Trade close | ±large | Core learning signal |
| 2 | Win bonus | Profitable close | +0.02 | Encourage winners |
| 3 | Smart HOLD | HOLD in position | -0.003 to +0.005 | Let winners run |
| 3b | Hold cost | Any step in position | -0.0002 | Prevent infinite hold |
| 4 | Spread | Trade open/flip | -0.025/-0.050 | Realistic cost |
| 5 | Inactivity | Flat >200 steps | 0 to -0.003 | Prevent idle |
| 6 | Churning | Avg trade <50 bars | 0 to -0.002 | Prevent overtrading |
| 7 | Drawdown | DD>5% + neg reward | ×1.5 | Risk management |

---

## 3. PPO Training Loop (Stable-Baselines3)

**File**: [trainer.py](file:///c:/Projects/Trading/XAU_LEarningModel/trainer.py)

### Single-Month Training

```python
model = PPO(
    "MlpPolicy",
    env=TradingEnv(month_data),
    learning_rate=3e-4,
    n_steps=2048,       # Rollout length before update
    batch_size=128,     # Minibatch for gradient step
    n_epochs=10,        # Passes over rollout buffer
    gamma=0.99,         # Discount factor
    ent_coef=0.05,      # Entropy bonus (exploration)
    clip_range=0.2,     # PPO clipping
    device="cuda",
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=torch.nn.ReLU,
    ),
)
model.learn(total_timesteps=20000)
```

### Parallel Multi-Month Training

```python
# One subprocess per month → SubprocVecEnv
env_fns = [make_env(chunk) for chunk in monthly_chunks]
vec_env = SubprocVecEnv(env_fns)      # Each month in own process
vec_env = VecMonitor(vec_env)          # Track rewards

model = PPO("MlpPolicy", vec_env, device="cuda", ...)
model.learn(total_timesteps=n_steps × n_envs)
```

### Network Architecture

```
Input (14-dim obs)
    ├─ Policy Net:  [256] → ReLU → [128] → ReLU → Softmax(4 actions)
    └─ Value Net:   [256] → ReLU → [128] → ReLU → Scalar (state value)
```

### Training Loop (per PPO update cycle)

```
1. Collect rollout:  env.step() × n_steps × n_envs
2. Compute GAE advantages using rewards + value estimates
3. For n_epochs:
   a. Shuffle rollout into minibatches (batch_size=128)
   b. Compute new policy π_new(a|s) and value V(s)
   c. Policy loss = -min(r·Â, clip(r, 1±ε)·Â)  where r = π_new/π_old
   d. Value loss = (V(s) - R_target)²
   e. Entropy bonus = H(π) × ent_coef
   f. Total loss = policy_loss + 0.5×value_loss - entropy_bonus
   g. Gradient step on GPU
4. Repeat from step 1
```

---

## 4. Observation Space (Feature Engineering)

**File**: [features.py](file:///c:/Projects/Trading/XAU_LEarningModel/rl_model/features.py)

```python
def precompute_features(df) → np.array(N, 12):
    """Pre-computed once per episode, indexed per step."""
    return [
        close/MA5 - 1,          # Short-term trend
        close/MA10 - 1,         # Medium-term trend  
        close/MA20 - 1,         # Trend strength
        close/MA100 - 1,        # Long-term bias
        (MA5-MA20)/close,       # Crossover signal
        (MA10-MA100)/close,     # Trend confirmation
        5-bar % return,         # Short momentum
        10-bar % return,        # Medium momentum
        20-bar % return,        # Longer momentum
        20-bar volatility,      # Risk gauge
        sin(2π×hour/24),        # Session timing
        cos(2π×hour/24),        # Session timing
    ]

# + 2 live features added per step:
#   [12] position_flag:    -1 / 0 / +1
#   [13] unrealized_pnl:   normalized to [-1, 1]
```
