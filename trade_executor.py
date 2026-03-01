"""
Trade Executor
===============
Handles order execution: buy, sell, close, modify positions.
All trades go through MT5's order_send with pre-flight checks.
"""

import logging
from datetime import datetime

import MetaTrader5 as mt5

import config

logger = logging.getLogger(__name__)


# ── Order Result ─────────────────────────────────

class OrderResult:
    """Structured result from an order attempt."""

    def __init__(self, success, ticket=None, price=None, volume=None,
                 action=None, symbol=None, sl=None, tp=None,
                 retcode=None, comment="", error=None):
        self.success = success
        self.ticket = ticket
        self.price = price
        self.volume = volume
        self.action = action
        self.symbol = symbol
        self.sl = sl
        self.tp = tp
        self.retcode = retcode
        self.comment = comment
        self.error = error
        self.timestamp = datetime.now()

    def __repr__(self):
        if self.success:
            return (f"OrderResult(OK, {self.action} {self.volume} {self.symbol} "
                    f"@ {self.price}, ticket={self.ticket})")
        return f"OrderResult(FAILED, {self.error}, retcode={self.retcode})"

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "action": self.action,
            "symbol": self.symbol,
            "ticket": self.ticket,
            "price": self.price,
            "volume": self.volume,
            "sl": self.sl,
            "tp": self.tp,
            "retcode": self.retcode,
            "comment": self.comment,
            "error": self.error,
        }


# ── Pre-flight Checks ───────────────────────────

def _preflight_check(symbol=None):
    """Validate trading conditions before sending an order."""
    symbol = symbol or config.SYMBOL

    # Check symbol exists and is tradeable
    info = mt5.symbol_info(symbol)
    if info is None:
        return False, f"Symbol {symbol} not found"

    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        if not info.visible:
            return False, f"Symbol {symbol} cannot be made visible"

    # Check spread
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, f"Cannot get tick for {symbol}"

    spread_points = round((tick.ask - tick.bid) / info.point)
    if spread_points > config.MAX_SPREAD_POINTS:
        return False, f"Spread too wide: {spread_points} points (max: {config.MAX_SPREAD_POINTS})"

    # Check max positions
    positions = mt5.positions_get(symbol=symbol)
    if positions is not None and len(positions) >= config.MAX_POSITIONS:
        return False, f"Max positions reached ({config.MAX_POSITIONS})"

    return True, "OK"


def _get_filling_type(symbol=None):
    """Determine the correct filling type for a symbol."""
    symbol = symbol or config.SYMBOL
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC

    filling = info.filling_mode
    if filling & mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    elif filling & mt5.SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN


# ── Order Execution ──────────────────────────────

def open_buy(symbol=None, lot=None, sl=None, tp=None, comment=None):
    """
    Open a BUY market order.

    Args:
        symbol:  Instrument (default: config.SYMBOL)
        lot:     Volume (default: config.LOT_SIZE)
        sl:      Stop loss price (None = no SL)
        tp:      Take profit price (None = no TP)
        comment: Order comment

    Returns:
        OrderResult
    """
    symbol = symbol or config.SYMBOL
    lot = lot or config.LOT_SIZE
    comment = comment or config.ORDER_COMMENT

    # Pre-flight
    ok, msg = _preflight_check(symbol)
    if not ok:
        logger.warning(f"BUY pre-flight failed: {msg}")
        return OrderResult(success=False, action="BUY", symbol=symbol, error=msg)

    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": config.MAX_SLIPPAGE,
        "magic": config.MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _get_filling_type(symbol),
    }

    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp

    result = mt5.order_send(request)
    return _process_result(result, "BUY", symbol, lot)


def open_sell(symbol=None, lot=None, sl=None, tp=None, comment=None):
    """Open a SELL market order."""
    symbol = symbol or config.SYMBOL
    lot = lot or config.LOT_SIZE
    comment = comment or config.ORDER_COMMENT

    ok, msg = _preflight_check(symbol)
    if not ok:
        logger.warning(f"SELL pre-flight failed: {msg}")
        return OrderResult(success=False, action="SELL", symbol=symbol, error=msg)

    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": config.MAX_SLIPPAGE,
        "magic": config.MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _get_filling_type(symbol),
    }

    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp

    result = mt5.order_send(request)
    return _process_result(result, "SELL", symbol, lot)


def close_position(ticket, symbol=None):
    """Close a specific position by its ticket number."""
    symbol = symbol or config.SYMBOL

    # Get the position details
    position = mt5.positions_get(ticket=ticket)
    if position is None or len(position) == 0:
        return OrderResult(success=False, action="CLOSE", symbol=symbol,
                           error=f"Position {ticket} not found")

    pos = position[0]
    tick = mt5.symbol_info_tick(pos.symbol)

    # Close = opposite order
    if pos.type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        close_price = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        close_price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": ticket,
        "price": close_price,
        "deviation": config.MAX_SLIPPAGE,
        "magic": config.MAGIC_NUMBER,
        "comment": f"CLOSE_{config.ORDER_COMMENT}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _get_filling_type(pos.symbol),
    }

    result = mt5.order_send(request)
    return _process_result(result, "CLOSE", pos.symbol, pos.volume)


def close_all_positions(symbol=None):
    """Close all open positions for a symbol."""
    symbol = symbol or config.SYMBOL
    positions = mt5.positions_get(symbol=symbol)

    if positions is None or len(positions) == 0:
        logger.info(f"No open positions to close for {symbol}")
        return []

    results = []
    for pos in positions:
        result = close_position(pos.ticket, symbol)
        results.append(result)
        if result.success:
            logger.info(f"Closed position {pos.ticket}: {result}")
        else:
            logger.error(f"Failed to close position {pos.ticket}: {result}")

    return results


def modify_position(ticket, sl=None, tp=None):
    """Modify SL/TP of an existing position."""
    position = mt5.positions_get(ticket=ticket)
    if position is None or len(position) == 0:
        return OrderResult(success=False, action="MODIFY",
                           error=f"Position {ticket} not found")

    pos = position[0]

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": ticket,
        "sl": sl if sl is not None else pos.sl,
        "tp": tp if tp is not None else pos.tp,
    }

    result = mt5.order_send(request)
    return _process_result(result, "MODIFY", pos.symbol, pos.volume)


# ── SL/TP Calculation ───────────────────────────

def calculate_sl_tp(action, current_price, ma_dict, symbol=None):
    """
    Calculate Stop Loss and Take Profit based on MA levels.

    Strategy:
        BUY:  SL = min(MA20, MA100) - buffer  |  TP = price + 2 * (price - SL)
        SELL: SL = max(MA20, MA100) + buffer  |  TP = price - 2 * (SL - price)

    Falls back to config defaults if MAs aren't available.

    Returns:
        (sl_price, tp_price)
    """
    symbol = symbol or config.SYMBOL
    info = mt5.symbol_info(symbol)
    point = info.point if info else 0.01  # XAUUSD point = $0.01

    ma20 = ma_dict.get("MA20")
    ma100 = ma_dict.get("MA100")

    if action == "BUY":
        if ma20 is not None and ma100 is not None and not any(map(lambda x: x != x, [ma20, ma100])):
            sl = min(ma20, ma100) - (20 * point)  # 20 points below lowest MA
        else:
            sl = current_price - (config.DEFAULT_SL_POINTS * point)

        risk = current_price - sl
        tp = current_price + (risk * 2)  # 2:1 reward/risk

    elif action == "SELL":
        if ma20 is not None and ma100 is not None and not any(map(lambda x: x != x, [ma20, ma100])):
            sl = max(ma20, ma100) + (20 * point)
        else:
            sl = current_price + (config.DEFAULT_SL_POINTS * point)

        risk = sl - current_price
        tp = current_price - (risk * 2)

    else:
        return None, None

    # Round to symbol digits
    digits = info.digits if info else 2
    sl = round(sl, digits)
    tp = round(tp, digits)

    logger.info(f"SL/TP for {action}: SL={sl}, TP={tp} (price={current_price})")
    return sl, tp


# ── Internal Helpers ─────────────────────────────

def _process_result(result, action, symbol, volume):
    """Convert MT5 order result to OrderResult."""
    if result is None:
        error = mt5.last_error()
        logger.error(f"Order {action} failed — result is None: {error}")
        return OrderResult(success=False, action=action, symbol=symbol, error=str(error))

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(
            f"Order {action} SUCCESS — ticket={result.order}, "
            f"price={result.price}, volume={result.volume}"
        )
        return OrderResult(
            success=True,
            ticket=result.order,
            price=result.price,
            volume=result.volume,
            action=action,
            symbol=symbol,
            retcode=result.retcode,
            comment=result.comment,
        )
    else:
        logger.error(
            f"Order {action} FAILED — retcode={result.retcode}, "
            f"comment={result.comment}"
        )
        return OrderResult(
            success=False,
            action=action,
            symbol=symbol,
            retcode=result.retcode,
            error=f"retcode={result.retcode}: {result.comment}",
        )


# ── Standalone test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from mt5_connection import MT5Connection

    print("Trade Executor — Interactive Test")
    print("=" * 50)

    with MT5Connection() as conn:
        print(conn.get_account_summary())
        print()

        # Show symbol info
        info = mt5.symbol_info(config.SYMBOL)
        if info:
            print(f"Symbol: {info.name}")
            print(f"Point: {info.point}")
            print(f"Digits: {info.digits}")
            print(f"Min Lot: {info.volume_min}")
            print(f"Spread: {info.spread} points")
            print()

        # Show current tick
        tick = mt5.symbol_info_tick(config.SYMBOL)
        if tick:
            print(f"Bid: {tick.bid}  |  Ask: {tick.ask}")
            print(f"Spread: {round((tick.ask - tick.bid) / info.point)} points")
            print()

        print("To test trading, uncomment the lines below in this script.")
        # Uncomment to test:
        # result = open_buy()
        # print(f"Buy result: {result}")
        # if result.success:
        #     import time; time.sleep(5)
        #     close_result = close_position(result.ticket)
        #     print(f"Close result: {close_result}")

    print("=" * 50)
    print("Trade Executor test complete.")
