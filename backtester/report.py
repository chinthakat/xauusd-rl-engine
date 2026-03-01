"""
Performance Report
===================
Generates formatted terminal report and HTML summary.
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)


def print_report(result):
    """Print a formatted performance report to terminal."""
    s = result.summary()

    print("\n" + "=" * 70)
    print("  BACKTEST PERFORMANCE REPORT")
    print("=" * 70)

    print(f"\n  {'Symbol:':<25} {result.config.symbol}")
    print(f"  {'Timeframe:':<25} {result.total_bars:,} bars")
    print(f"  {'Duration:':<25} {s['duration_sec']:.1f}s ({s['bars_per_sec']:.0f} bars/sec)")

    print(f"\n  ── Account {'─'*44}")
    print(f"  {'Initial Balance:':<25} ${s['initial_balance']:>10,.2f}")
    print(f"  {'Final Balance:':<25} ${s['final_balance']:>10,.2f}")
    print(f"  {'Net Profit:':<25} ${s['net_profit']:>+10,.2f}")
    print(f"  {'Return:':<25} {s['return_pct']:>+10.2f}%")
    print(f"  {'Max Drawdown:':<25} {s['max_drawdown_pct']:>10.2f}%")
    print(f"  {'Sharpe Ratio:':<25} {s['sharpe_ratio']:>10.4f}")

    print(f"\n  ── Trades {'─'*45}")
    print(f"  {'Total Trades:':<25} {s['total_trades']:>10d}")
    print(f"  {'Win / Loss / BE:':<25} {s['winning_trades']:>3d} / {s['losing_trades']:>3d} / {s['breakeven_trades']:>3d}")
    print(f"  {'Win Rate:':<25} {s['win_rate']:>10.2f}%")
    print(f"  {'Profit Factor:':<25} {s['profit_factor']:>10.4f}")
    print(f"  {'Avg Win:':<25} ${s['avg_profit']:>+10.2f}")
    print(f"  {'Avg Loss:':<25} ${s['avg_loss']:>+10.2f}")
    print(f"  {'Largest Win:':<25} ${s['largest_win']:>+10.2f}")
    print(f"  {'Largest Loss:':<25} ${s['largest_loss']:>+10.2f}")

    print(f"\n  ── Costs {'─'*46}")
    print(f"  {'Total Commission:':<25} ${s['total_commission']:>10.2f}")
    print(f"  {'Total Swap:':<25} ${s['total_swap']:>+10.2f}")
    print(f"  {'Margin Calls:':<25} {s['margin_calls']:>10d}")
    print(f"  {'Stop Outs:':<25} {s['stop_outs']:>10d}")

    print(f"\n  ── Settings {'─'*43}")
    c = result.config
    print(f"  {'Spread:':<25} {c.spread_points} pts ({'ON' if c.spread_enabled else 'OFF'})")
    print(f"  {'Slippage:':<25} {c.slippage_points[0]}-{c.slippage_points[1]} pts ({'ON' if c.slippage_enabled else 'OFF'})")
    print(f"  {'Latency:':<25} {c.latency_ms[0]}-{c.latency_ms[1]} ms ({'ON' if c.latency_enabled else 'OFF'})")
    print(f"  {'Commission:':<25} ${c.commission_per_lot}/lot ({'ON' if c.commission_enabled else 'OFF'})")
    print(f"  {'Swap:':<25} L:{c.swap_long} / S:{c.swap_short} ({'ON' if c.swap_enabled else 'OFF'})")
    print(f"  {'Margin:':<25} 1:{c.leverage} ({'ON' if c.margin_enabled else 'OFF'})")
    print(f"  {'Tick Mode:':<25} {c.tick_mode}")

    print("=" * 70)


def print_trade_log(result, max_trades=20):
    """Print recent trades in a table format."""
    trades = result.trades
    if not trades:
        print("\n  No trades to display.")
        return

    show = trades[-max_trades:]
    print(f"\n  ── Last {len(show)} Trades {'─'*39}")
    print(f"  {'#':>4}  {'Type':<5} {'Entry':>9} {'Exit':>9} {'P/L':>9} {'Comm':>7} {'Swap':>7}")
    print(f"  {'─'*4}  {'─'*5} {'─'*9} {'─'*9} {'─'*9} {'─'*7} {'─'*7}")

    for t in show:
        pnl_str = f"${t.profit:+.2f}"
        print(f"  {t.ticket:>4}  {t.type:<5} {t.entry_price:>9.2f} {t.exit_price:>9.2f} "
              f"{pnl_str:>9} ${t.commission:>6.2f} ${t.swap:>6.2f}")

    if len(trades) > max_trades:
        print(f"  ... ({len(trades) - max_trades} more trades)")


def export_trades_csv(result, filepath):
    """Export trade history to CSV."""
    import csv
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ticket", "type", "entry_price", "exit_price", "lots",
                         "profit", "commission", "swap", "slippage", "comment"])
        for t in result.trades:
            writer.writerow([t.ticket, t.type, t.entry_price, t.exit_price,
                             t.lots, t.profit, t.commission, t.swap,
                             t.slippage, t.comment])

    logger.info(f"Trades exported to {filepath}")
