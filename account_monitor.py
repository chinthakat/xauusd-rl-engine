"""
Account Monitor
================
Tracks account balance, equity, open positions, and trade history.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import MetaTrader5 as mt5

import config

logger = logging.getLogger(__name__)


# ── Account Info ─────────────────────────────────

def get_account_info():
    """
    Get current account snapshot.

    Returns:
        dict with balance, equity, margin, free_margin, profit, currency, leverage
    """
    info = mt5.account_info()
    if info is None:
        logger.error(f"Failed to get account info: {mt5.last_error()}")
        return None

    return {
        "login": info.login,
        "server": info.server,
        "currency": info.currency,
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "margin_level": info.margin_level,
        "profit": info.profit,
        "leverage": info.leverage,
        "timestamp": datetime.now().isoformat(),
    }


# ── Position Monitoring ─────────────────────────

def get_open_positions(symbol=None):
    """
    Get all open positions as a DataFrame.

    Returns:
        pd.DataFrame with columns: ticket, symbol, type, volume, price_open,
        sl, tp, profit, swap, time, comment
    """
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()

    if positions is None or len(positions) == 0:
        return pd.DataFrame()

    data = []
    for pos in positions:
        data.append({
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume": pos.volume,
            "price_open": pos.price_open,
            "price_current": pos.price_current,
            "sl": pos.sl,
            "tp": pos.tp,
            "profit": pos.profit,
            "swap": pos.swap,
            "time": datetime.fromtimestamp(pos.time),
            "magic": pos.magic,
            "comment": pos.comment,
        })

    return pd.DataFrame(data)


def has_open_position(symbol=None):
    """Check if there's an open position for the symbol."""
    symbol = symbol or config.SYMBOL
    positions = mt5.positions_get(symbol=symbol)
    return positions is not None and len(positions) > 0


def get_position_type(symbol=None):
    """Get the type of the current open position ('BUY', 'SELL', or None)."""
    symbol = symbol or config.SYMBOL
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return None
    return "BUY" if positions[0].type == mt5.ORDER_TYPE_BUY else "SELL"


def get_position_profit(ticket=None, symbol=None):
    """Get the P/L of a specific position or the first position for the symbol."""
    if ticket:
        positions = mt5.positions_get(ticket=ticket)
    else:
        symbol = symbol or config.SYMBOL
        positions = mt5.positions_get(symbol=symbol)

    if positions is None or len(positions) == 0:
        return 0.0

    return positions[0].profit


# ── Trade History ────────────────────────────────

def get_trade_history(symbol=None, days=30):
    """
    Fetch closed trade history (deals) for a symbol.

    Returns:
        pd.DataFrame with deal details
    """
    symbol = symbol or config.SYMBOL
    end = datetime.now()
    start = end - timedelta(days=days)

    deals = mt5.history_deals_get(start, end, group=f"*{symbol}*")

    if deals is None or len(deals) == 0:
        logger.info(f"No trade history found for {symbol} in last {days} days")
        return pd.DataFrame()

    data = []
    for deal in deals:
        data.append({
            "ticket": deal.ticket,
            "order": deal.order,
            "time": datetime.fromtimestamp(deal.time),
            "type": deal.type,
            "entry": deal.entry,  # 0=in, 1=out
            "symbol": deal.symbol,
            "volume": deal.volume,
            "price": deal.price,
            "profit": deal.profit,
            "swap": deal.swap,
            "commission": deal.commission,
            "fee": deal.fee,
            "magic": deal.magic,
            "comment": deal.comment,
        })

    return pd.DataFrame(data)


def get_order_history(symbol=None, days=30):
    """Fetch order history for a symbol."""
    symbol = symbol or config.SYMBOL
    end = datetime.now()
    start = end - timedelta(days=days)

    orders = mt5.history_orders_get(start, end, group=f"*{symbol}*")

    if orders is None or len(orders) == 0:
        return pd.DataFrame()

    data = []
    for order in orders:
        data.append({
            "ticket": order.ticket,
            "time_setup": datetime.fromtimestamp(order.time_setup),
            "time_done": datetime.fromtimestamp(order.time_done) if order.time_done else None,
            "type": order.type,
            "state": order.state,
            "symbol": order.symbol,
            "volume_initial": order.volume_initial,
            "volume_current": order.volume_current,
            "price_open": order.price_open,
            "sl": order.sl,
            "tp": order.tp,
            "magic": order.magic,
            "comment": order.comment,
        })

    return pd.DataFrame(data)


# ── Standalone test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from mt5_connection import MT5Connection

    print("Account Monitor — Test")
    print("=" * 50)

    with MT5Connection() as conn:
        # Account info
        acct = get_account_info()
        if acct:
            print("\nAccount Info:")
            for k, v in acct.items():
                print(f"  {k}: {v}")

        # Open positions
        positions = get_open_positions()
        print(f"\nOpen Positions: {len(positions)}")
        if not positions.empty:
            print(positions.to_string())

        # Trade history
        history = get_trade_history(days=7)
        print(f"\nTrade History (7 days): {len(history)} deals")
        if not history.empty:
            print(history.tail(5).to_string())

    print("\n" + "=" * 50)
    print("Account Monitor test complete.")
