"""
Configuration Template
=======================
Copy this file to `config.py` and fill in your own values.

`config.py` is git-ignored on purpose — it holds MT5 account credentials.
Nine modules import it directly as `import config` (main, data_manager,
mt5_connection, trade_executor, account_monitor, trade_logger, rl_model/train,
rl_model/evaluate and tests/test_data_manager); the backtester package picks it
up transitively through data_manager. The file must exist at the repository root
before anything useful will run.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── MetaTrader 5 Terminal / Account ───────────────
# Use a DEMO account. See the risk notice in README.md.
MT5_ACCOUNT = 12345678                     # int — MT5 login number
MT5_PASSWORD = "your-mt5-password"         # str — MT5 account password
MT5_SERVER = "YourBroker-Demo"             # str — broker server name
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # str or None to auto-detect

# ── Instrument ────────────────────────────────────
SYMBOL = "XAUUSD"                          # Symbol name as your broker spells it
LOT_SIZE = 0.01                            # Volume per trade

# ── Order Handling ────────────────────────────────
MAGIC_NUMBER = 888888                      # Identifies this system's orders in MT5
ORDER_COMMENT = "XAU_RL"                   # Comment written on each order
MAX_SLIPPAGE = 30                          # Max price deviation, in points
MAX_SPREAD_POINTS = 50                     # Refuse to open if spread exceeds this
MAX_POSITIONS = 1                          # Max simultaneous open positions
DEFAULT_SL_POINTS = 500                    # Fallback stop-loss distance, in points

# ── Strategy / Data ───────────────────────────────
MA_PERIODS = [5, 10, 20, 100]              # Moving-average windows to compute
INITIAL_BARS = 200                         # M1 bars fetched per live-loop poll
LOOP_INTERVAL_SECONDS = 60                 # Seconds between live-loop iterations

# ── Paths ─────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "Datafiles")   # Folder holding XAUUSD*.csv history
LOG_DIR = os.path.join(BASE_DIR, "logs")

SYSTEM_LOG_FILE = os.path.join(LOG_DIR, "system.log")
TRADE_LOG_FILE = os.path.join(LOG_DIR, "trades.csv")
ACCOUNT_LOG_FILE = os.path.join(LOG_DIR, "account.csv")
