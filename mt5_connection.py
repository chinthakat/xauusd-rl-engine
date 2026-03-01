"""
MT5 Connection Manager
=======================
Handles MetaTrader 5 terminal initialization, login, shutdown,
and connection health monitoring.
"""

import time
import logging
import MetaTrader5 as mt5

import config

logger = logging.getLogger(__name__)


class MT5Connection:
    """Context-manager wrapper for MT5 terminal connection."""

    def __init__(self, account=None, password=None, server=None, mt5_path=None):
        self.account = account or config.MT5_ACCOUNT
        self.password = password or config.MT5_PASSWORD
        self.server = server or config.MT5_SERVER
        self.mt5_path = mt5_path or config.MT5_PATH
        self._connected = False

    # ── Context manager ──────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    # ── Core methods ─────────────────────────────

    def connect(self, max_retries=3, retry_delay=2.0):
        """Initialize MT5 terminal and login. Retries with exponential backoff."""
        for attempt in range(1, max_retries + 1):
            try:
                # Initialize terminal
                init_kwargs = {}
                if self.mt5_path:
                    init_kwargs["path"] = self.mt5_path

                if not mt5.initialize(**init_kwargs):
                    error = mt5.last_error()
                    logger.warning(f"MT5 init failed (attempt {attempt}/{max_retries}): {error}")
                    if attempt < max_retries:
                        time.sleep(retry_delay * attempt)
                        continue
                    raise ConnectionError(f"MT5 initialization failed after {max_retries} attempts: {error}")

                # Login to account
                if self.account and self.password and self.server:
                    if not mt5.login(self.account, password=self.password, server=self.server):
                        error = mt5.last_error()
                        mt5.shutdown()
                        logger.warning(f"MT5 login failed (attempt {attempt}/{max_retries}): {error}")
                        if attempt < max_retries:
                            time.sleep(retry_delay * attempt)
                            continue
                        raise ConnectionError(f"MT5 login failed after {max_retries} attempts: {error}")

                self._connected = True
                account_info = mt5.account_info()
                logger.info(
                    f"Connected to MT5 — Account: {account_info.login}, "
                    f"Server: {account_info.server}, "
                    f"Balance: {account_info.balance:.2f} {account_info.currency}"
                )
                return True

            except ConnectionError:
                raise
            except Exception as e:
                logger.error(f"Unexpected error during connection (attempt {attempt}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    continue
                raise

    def disconnect(self):
        """Cleanly shut down MT5 connection."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 connection closed")

    def is_connected(self):
        """Check if terminal is alive and account is accessible."""
        if not self._connected:
            return False
        try:
            info = mt5.account_info()
            return info is not None
        except Exception:
            return False

    def reconnect(self):
        """Drop and re-establish the connection."""
        logger.info("Reconnecting to MT5...")
        self.disconnect()
        time.sleep(1)
        return self.connect()

    # ── Info helpers ─────────────────────────────

    def get_terminal_info(self):
        """Return terminal version and connection details."""
        if not self.is_connected():
            return {"status": "disconnected"}

        term = mt5.terminal_info()
        acct = mt5.account_info()
        return {
            "status": "connected",
            "terminal_build": term.build if term else None,
            "terminal_path": term.path if term else None,
            "account_login": acct.login if acct else None,
            "account_server": acct.server if acct else None,
            "account_balance": acct.balance if acct else None,
            "account_currency": acct.currency if acct else None,
            "account_leverage": acct.leverage if acct else None,
            "trade_allowed": term.trade_allowed if term else None,
        }

    def get_account_summary(self):
        """Quick account summary string."""
        info = self.get_terminal_info()
        if info["status"] == "disconnected":
            return "MT5: Disconnected"
        return (
            f"MT5: {info['account_login']}@{info['account_server']} | "
            f"Balance: {info['account_balance']:.2f} {info['account_currency']} | "
            f"Leverage: 1:{info['account_leverage']} | "
            f"Trading: {'Enabled' if info['trade_allowed'] else 'DISABLED'}"
        )


# ── Standalone test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Testing MT5 Connection...")
    print("-" * 50)

    with MT5Connection() as conn:
        print(conn.get_account_summary())
        print()
        info = conn.get_terminal_info()
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()
        print(f"  is_connected: {conn.is_connected()}")

    print("-" * 50)
    print("Connection test complete.")
