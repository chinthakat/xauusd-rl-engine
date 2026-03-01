"""
Interactive Visualizer
=======================
Plotly-based interactive charting with candlesticks, trade markers,
indicators, and equity curve. Opens in browser.
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("Plotly not installed. Run: pip install plotly")


def create_chart(result, output_path=None, show=True):
    """
    Create an interactive chart from backtest results.

    Args:
        result:      BacktestResult from engine.run()
        output_path: Save HTML to file (optional)
        show:        Open in browser

    Returns:
        plotly Figure
    """
    if not HAS_PLOTLY:
        print("ERROR: Plotly is required. Install: pip install plotly")
        return None

    df = result.ohlc_data
    trades = result.trades
    summary = result.summary()

    # ── Create subplot layout ────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.25, 0.15],
        subplot_titles=("", "Equity & Balance", "Volume"),
    )

    # ── Panel 1: Candlestick + MAs + Trade Markers ──

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
        opacity=0.9,
    ), row=1, col=1)

    # Moving Averages
    ma_colors = {
        "MA5": "#FF9800",   # Orange
        "MA10": "#2196F3",  # Blue
        "MA20": "#9C27B0",  # Purple
        "MA100": "#FFFFFF", # White
    }
    for ma_name, color in ma_colors.items():
        if ma_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[ma_name],
                name=ma_name,
                line=dict(color=color, width=1),
                opacity=0.7,
                visible="legendonly" if ma_name == "MA100" else True,
            ), row=1, col=1)

    # Trade markers
    buy_times, buy_prices, buy_texts = [], [], []
    sell_times, sell_prices, sell_texts = [], [], []
    close_times, close_prices, close_texts = [], [], []

    for ts, price, marker_type, ticket in result.trade_markers:
        dt = datetime.fromtimestamp(ts) if isinstance(ts, (int, float)) else ts
        trade = next((t for t in trades if t.ticket == ticket), None)

        if marker_type == "buy":
            buy_times.append(dt)
            buy_prices.append(price)
            buy_texts.append(f"BUY #{ticket}<br>@ {price:.2f}")
        elif marker_type == "sell":
            sell_times.append(dt)
            sell_prices.append(price)
            sell_texts.append(f"SELL #{ticket}<br>@ {price:.2f}")
        elif marker_type.startswith("close") or marker_type == "stop_out":
            close_times.append(dt)
            close_prices.append(price)
            pnl_str = f"${trade.profit:+.2f}" if trade else ""
            label = "STOP OUT" if marker_type == "stop_out" else "CLOSE"
            close_texts.append(f"{label} #{ticket}<br>@ {price:.2f}<br>P/L: {pnl_str}")

    # Buy markers
    if buy_times:
        fig.add_trace(go.Scatter(
            x=buy_times, y=buy_prices,
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", size=12, color="#00E676",
                       line=dict(color="#004D40", width=1)),
            text=buy_texts,
            hoverinfo="text",
        ), row=1, col=1)

    # Sell markers
    if sell_times:
        fig.add_trace(go.Scatter(
            x=sell_times, y=sell_prices,
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", size=12, color="#FF1744",
                       line=dict(color="#B71C1C", width=1)),
            text=sell_texts,
            hoverinfo="text",
        ), row=1, col=1)

    # Close markers
    if close_times:
        fig.add_trace(go.Scatter(
            x=close_times, y=close_prices,
            mode="markers",
            name="Close",
            marker=dict(symbol="x", size=10, color="#FFC107",
                       line=dict(color="#FF6F00", width=2)),
            text=close_texts,
            hoverinfo="text",
        ), row=1, col=1)

    # ── Panel 2: Equity & Balance Curve ──────

    if result.equity_curve:
        eq_df = pd.DataFrame(result.equity_curve)
        eq_df["datetime"] = pd.to_datetime(eq_df["timestamp"], unit="s")

        # Balance line
        fig.add_trace(go.Scatter(
            x=eq_df["datetime"],
            y=eq_df["balance"],
            name="Balance",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        ), row=2, col=1)

        # Equity line
        fig.add_trace(go.Scatter(
            x=eq_df["datetime"],
            y=eq_df["equity"],
            name="Equity",
            line=dict(color="#FF9800", width=1.5),
        ), row=2, col=1)

        # Drawdown shading
        peak = eq_df["equity"].expanding().max()
        drawdown = (peak - eq_df["equity"]) / peak * 100
        dd_mask = drawdown > 2  # Show drawdown > 2%
        if dd_mask.any():
            fig.add_trace(go.Scatter(
                x=eq_df["datetime"][dd_mask],
                y=eq_df["equity"][dd_mask],
                name="Drawdown",
                line=dict(color="rgba(255,0,0,0)", width=0),
                fill="tonexty",
                fillcolor="rgba(244,67,54,0.2)",
                showlegend=False,
            ), row=2, col=1)

    # ── Panel 3: Volume ──────────────────────

    if "Volume" in df.columns:
        colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors.tolist(),
            opacity=0.6,
            showlegend=False,
        ), row=3, col=1)

    # ── Layout & Styling ────────────────────

    # Performance stats header
    title_text = (
        f"<b>{result.config.symbol} Backtest</b> — "
        f"Net: <b>${summary['net_profit']:+,.2f}</b> ({summary['return_pct']:+.2f}%) | "
        f"Trades: {summary['total_trades']} | "
        f"Win: {summary['win_rate']:.1f}% | "
        f"PF: {summary['profit_factor']:.2f} | "
        f"Max DD: {summary['max_drawdown_pct']:.1f}% | "
        f"Sharpe: {summary['sharpe_ratio']:.2f}"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=14)),
        template="plotly_dark",
        height=900,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=40, t=80, b=40),
        hovermode="x unified",
        dragmode="zoom",
    )

    # Y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="$", row=2, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Vol", row=3, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

    # Save & show
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "backtest_chart.html")

    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    logger.info(f"Chart saved to: {output_path}")

    if show:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(output_path))

    return fig
