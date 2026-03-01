"""
MT5 Bridge — File-Based IPC with MQL5 EA
==========================================
Communicates with RLBridge.mq5 running in MT5 Strategy Tester
via file-based messaging in Common\\Files\\rl_bridge\\.

Protocol:
    1. EA writes state.csv (market state on each new bar)
    2. Python reads state.csv, computes action
    3. Python writes action.csv
    4. EA reads action, executes, writes result.csv
    5. Python reads result.csv (reward/done)
"""

import os
import time
import csv
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Default path to MT5 Common Files
DEFAULT_COMMON_FILES = os.path.join(
    os.environ.get("APPDATA", ""),
    "MetaQuotes", "Terminal", "Common", "Files"
)


class MT5Bridge:
    """
    File-based bridge between Python and MQL5 EA (RLBridge.mq5).
    Handles reading state, writing actions, and reading results.
    """

    def __init__(self, bridge_folder="rl_bridge", common_files_path=None,
                 timeout_sec=10.0, poll_interval_sec=0.001):
        """
        Args:
            bridge_folder:     Subfolder name in Common\\Files
            common_files_path: Path to MT5 Common\\Files (auto-detected)
            timeout_sec:       Max wait for EA response
            poll_interval_sec: File poll interval (1ms default)
        """
        if common_files_path is None:
            common_files_path = DEFAULT_COMMON_FILES

        self.bridge_dir = os.path.join(common_files_path, bridge_folder)
        self.timeout = timeout_sec
        self.poll_interval = poll_interval_sec

        # File paths
        self.state_file = os.path.join(self.bridge_dir, "state.csv")
        self.action_file = os.path.join(self.bridge_dir, "action.csv")
        self.result_file = os.path.join(self.bridge_dir, "result.csv")
        self.reset_file = os.path.join(self.bridge_dir, "reset.csv")
        self.ready_file = os.path.join(self.bridge_dir, "ready.csv")

        # Ensure bridge directory exists
        os.makedirs(self.bridge_dir, exist_ok=True)

        # Track state file modification time to detect new bars
        self._last_state_mtime = 0

        logger.info(f"MT5 Bridge initialized: {self.bridge_dir}")

    def wait_for_ea_ready(self, timeout=30):
        """Wait for the EA to signal it's ready."""
        logger.info("Waiting for RLBridge EA to start...")
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self.ready_file):
                logger.info("EA is ready!")
                return True
            time.sleep(0.1)
        logger.error(f"EA not ready after {timeout}s. Is MT5 Strategy Tester running?")
        return False

    def send_reset(self):
        """Signal the EA to reset for a new episode."""
        # Write reset file
        with open(self.reset_file, 'w') as f:
            f.write("reset\n")

        # Clean stale files
        self._safe_delete(self.state_file)
        self._safe_delete(self.action_file)
        self._safe_delete(self.result_file)

        self._last_state_mtime = 0
        logger.debug("Reset signal sent")

    def wait_for_state(self):
        """
        Wait for EA to write a new state file.

        Returns:
            dict with market state, or None on timeout
        """
        start = time.time()
        while time.time() - start < self.timeout:
            if os.path.exists(self.state_file):
                try:
                    mtime = os.path.getmtime(self.state_file)
                    if mtime > self._last_state_mtime:
                        state = self._read_csv(self.state_file)
                        if state:
                            self._last_state_mtime = mtime
                            return self._parse_state(state)
                except (OSError, PermissionError):
                    pass  # File still being written

            time.sleep(self.poll_interval)

        return None  # Timeout

    def send_action(self, action):
        """
        Write action for the EA to read.

        Args:
            action: int (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE)
        """
        # Delete old result file first
        self._safe_delete(self.result_file)

        with open(self.action_file, 'w') as f:
            f.write(str(int(action)))

    def wait_for_result(self):
        """
        Wait for EA to write execution result.

        Returns:
            dict with result, or None on timeout
        """
        start = time.time()
        while time.time() - start < self.timeout:
            if os.path.exists(self.result_file):
                try:
                    result = self._read_csv(self.result_file)
                    if result:
                        # Delete to avoid re-reading
                        self._safe_delete(self.result_file)
                        return self._parse_result(result)
                except (OSError, PermissionError):
                    pass

            time.sleep(self.poll_interval)

        return None  # Timeout

    def step(self, action):
        """
        Complete one step: send action, wait for result.

        Args:
            action: int (0-3)

        Returns:
            dict with 'action', 'fill_price', 'realized_pnl',
                       'balance', 'equity', 'position', 'done'
        """
        self.send_action(action)
        return self.wait_for_result()

    def cleanup(self):
        """Remove bridge files."""
        for f in [self.state_file, self.action_file, self.result_file,
                  self.reset_file, self.ready_file]:
            self._safe_delete(f)

    # ── Internal helpers ─────────────────────────

    def _read_csv(self, filepath):
        """Read a 2-row CSV (header + data) into a dict."""
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) >= 2:
                    headers = [h.strip() for h in rows[0]]
                    values = [v.strip() for v in rows[1]]
                    return dict(zip(headers, values))
        except Exception:
            return None
        return None

    def _parse_state(self, raw):
        """Parse state dict from CSV strings to proper types."""
        try:
            return {
                "timestamp": raw.get("timestamp", ""),
                "open": float(raw.get("open", 0)),
                "high": float(raw.get("high", 0)),
                "low": float(raw.get("low", 0)),
                "close": float(raw.get("close", 0)),
                "volume": int(raw.get("volume", 0)),
                "bid": float(raw.get("bid", 0)),
                "ask": float(raw.get("ask", 0)),
                "spread": float(raw.get("spread", 0)),
                "position": int(raw.get("position", 0)),
                "entry_price": float(raw.get("entry_price", 0)),
                "unrealized_pnl": float(raw.get("unrealized_pnl", 0)),
                "balance": float(raw.get("balance", 0)),
                "equity": float(raw.get("equity", 0)),
                "bar_count": int(raw.get("bar_count", 0)),
            }
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse state: {e}")
            return None

    def _parse_result(self, raw):
        """Parse result dict from CSV strings to proper types."""
        try:
            return {
                "action": int(raw.get("action", 0)),
                "fill_price": float(raw.get("fill_price", 0)),
                "realized_pnl": float(raw.get("realized_pnl", 0)),
                "balance": float(raw.get("balance", 0)),
                "equity": float(raw.get("equity", 0)),
                "position": int(raw.get("position", 0)),
                "done": raw.get("done", "0") == "1",
            }
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse result: {e}")
            return None

    def _safe_delete(self, filepath):
        """Delete a file, ignoring errors."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass
