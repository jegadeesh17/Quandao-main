"""
quandao_public/execution/order_executor.py
=============================================
Paper trading execution engine with Transaction Cost Analysis (TCA).

WHAT THIS MODULE DOES:
    Simulates order execution without calling any live broker API.
    Every "trade" is logged with full detail for post-analysis.
    TCA (Transaction Cost Analysis) decomposes P&L into:
        signal_alpha - explicit_costs - slippage - market_impact = net_pnl

DRY RUN GUARANTEE:
    DRY_RUN is always True (imported from config).
    The function place_order_dry_run() is the ONLY execution function.
    It LOGS the order and NEVER calls the Fyers API.
    This ensures you can run the full execution pipeline safely.

SIMULATED CAPITAL:
    Equity/NSE: ₹1,00,000 (SIMULATED_CAPITAL_INR)
    Forex/MT5:  $1,000    (SIMULATED_CAPITAL_USD)

PAPER TRADE LOG FORMAT:
    All trades are appended to quandao_private/results/live/paper_trades.json
    as a JSON array. Each entry is one complete trade (entry OR exit OR combined).

USAGE:
    from quandao_public.execution.order_executor import place_order_dry_run, run_tca_simulation

    # Simulate placing a buy order
    order = place_order_dry_run(
        symbol='NSE:NIFTY50-INDEX', side='BUY', quantity=75,
        price=24500.0, order_type='MARKET',
        strategy='multi_factor', signal_score=0.75,
    )
    print(order['order_id'])   # Simulated order ID
    print(order['status'])     # 'SIMULATED'
"""

import json
import logging
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from quandao_public.config import (
    DRY_RUN,
    SIMULATED_CAPITAL_INR,
    SIMULATED_CAPITAL_USD,
    PAPER_TRADE_LOG_PATH,
)
from quandao_public.strategies.cost_model import compute_round_trip_cost


# ── Constants ────────────────────────────────────────────────────────────────
_LOG_PATH = Path(PAPER_TRADE_LOG_PATH)
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Dry-Run Order Placement
# ══════════════════════════════════════════════════════════════════════════════

def place_order_dry_run(symbol: str,
                         side: str,
                         quantity: int,
                         price: float,
                         order_type: str = 'MARKET',
                         stop_loss: float = None,
                         target: float = None,
                         strategy: str = 'unknown',
                         signal_score: float = 0.0,
                         instrument: str = 'equity',
                         slippage_pts: float = 2.0,
                         notes: str = '') -> dict:
    """
    Simulate order placement without calling any live broker API.

    WHAT IT DOES:
        1. Validates the order parameters
        2. Applies realistic slippage to the fill price
        3. Computes explicit transaction costs (STT, brokerage, etc.)
        4. Generates a simulated order ID and response
        5. Logs the complete order to paper_trades.json

    SLIPPAGE MODEL:
        For MARKET orders: assumed fill = price ± slippage_pts
            BUY:  fill_price = price + slippage_pts  (you buy at the ask)
            SELL: fill_price = price - slippage_pts  (you sell at the bid)
        For LIMIT orders: fill_price = limit_price (no slippage, may not fill)

    ALWAYS DRY RUN:
        DRY_RUN = True is hardcoded in config.py.
        This function will never route to Fyers even if that code exists.

    Parameters
    ----------
    symbol       : str   - Instrument symbol (e.g. 'NSE:NIFTY50-INDEX')
    side         : str   - 'BUY' or 'SELL'
    quantity     : int   - Number of units (shares, lots * lot_size)
    price        : float - Intended price (mid-market reference)
    order_type   : str   - 'MARKET' or 'LIMIT'
    stop_loss    : float - Stop loss price (for record-keeping, not enforced)
    target       : float - Profit target price (for record-keeping)
    strategy     : str   - Strategy name that generated this order
    signal_score : float - Factor composite score at time of signal (-1 to +1)
    instrument   : str   - 'equity', 'futures', or 'options' (affects cost calc)
    slippage_pts : float - Expected bid-ask slippage (index points or price units)
    notes        : str   - Free-text notes (e.g. 'Factor momentum breakout')

    Returns
    -------
    dict : Simulated order response with full detail.
    """
    assert DRY_RUN, "DRY_RUN must be True. Live order placement not implemented."

    side = side.upper()
    ts   = datetime.now(timezone.utc).isoformat()

    # Slippage-adjusted fill price
    if order_type == 'MARKET':
        fill_price = price + slippage_pts if side == 'BUY' else price - slippage_pts
    else:
        fill_price = price   # Limit order: assume exact fill at limit

    fill_price = max(0.01, fill_price)  # Sanity check

    # Simulated order response
    order = {
        'order_id':      str(uuid.uuid4())[:8].upper(),  # Short 8-char ID
        'timestamp_utc': ts,
        'status':        'SIMULATED',
        'dry_run':       True,

        # Order details
        'symbol':        symbol,
        'side':          side,
        'quantity':      quantity,
        'order_type':    order_type,
        'intended_price': price,
        'fill_price':    round(fill_price, 2),
        'slippage_pts':  slippage_pts if order_type == 'MARKET' else 0.0,

        # Cost estimate (single leg only — for round-trip, use compute_round_trip_cost)
        'est_leg_cost_inr': _estimate_single_leg_cost(fill_price, quantity, side, instrument),

        # Risk parameters
        'stop_loss':     stop_loss,
        'target':        target,

        # Strategy metadata
        'strategy':      strategy,
        'signal_score':  round(signal_score, 4),
        'instrument':    instrument,
        'notes':         notes,
    }

    # Persist to log
    log_trade(order)

    _print_order_summary(order)
    return order


def _estimate_single_leg_cost(price: float, qty: int,
                                side: str, instrument: str) -> float:
    """Estimate single-leg transaction cost in ₹."""
    try:
        from quandao_public.strategies.cost_model import compute_leg_cost
        leg = compute_leg_cost(price, qty, side, instrument)
        return leg['total_inr']
    except Exception:
        return round(price * qty * 0.0003, 2)  # Rough 0.03% fallback


def _print_order_summary(order: dict) -> None:
    """Log a clean order summary at INFO level."""
    logger.info(
        "[DRY RUN] %s %s x%s @ %.2f | slippage=%.1fpts | "
        "strategy=%s signal=%.3f | leg_cost=Rs %.2f",
        order['side'],
        order['symbol'],
        order['quantity'],
        order['fill_price'],
        order['slippage_pts'],
        order['strategy'],
        order['signal_score'],
        order['est_leg_cost_inr'],
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Trade Logging
# ══════════════════════════════════════════════════════════════════════════════

def log_trade(trade_dict: dict, log_path: str = None) -> None:
    """
    Append a trade record to the paper trade log (JSON array file).

    The log file grows linearly — one entry per order leg.
    Use load_trade_log() to read all trades back as a DataFrame.

    Parameters
    ----------
    trade_dict : dict - Complete order/trade dictionary
    log_path   : str  - Override default log path (useful for testing)
    """
    path = Path(log_path) if log_path else _LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing log
    existing = []
    if path.exists():
        try:
            with open(path, 'r') as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)
        except json.JSONDecodeError:
            existing = []   # File was corrupted — start fresh

    # Append new trade
    existing.append(trade_dict)

    # Write back (pretty-printed for human readability)
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)


def load_trade_log(log_path: str = None) -> pd.DataFrame:
    """
    Load all paper trades from the log file into a DataFrame.

    Returns
    -------
    pd.DataFrame : All logged trades. Empty DataFrame if log doesn't exist.
    """
    path = Path(log_path) if log_path else _LOG_PATH

    if not path.exists():
        return pd.DataFrame()

    try:
        with open(path, 'r') as f:
            trades = json.load(f)
        return pd.DataFrame(trades)
    except Exception as e:
        print(f"[load_trade_log] Error reading log: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Transaction Cost Analysis (TCA)
# ══════════════════════════════════════════════════════════════════════════════

def run_tca_simulation(trades_df: pd.DataFrame,
                        market_data_df: pd.DataFrame = None) -> dict:
    """
    Transaction Cost Analysis (TCA) — decompose P&L into its cost components.

    TCA DECOMPOSITION:
        Gross P&L (signal alpha) = exit_price - entry_price (before costs)
        Slippage cost            = intended_price - fill_price (bid-ask spread)
        Explicit cost            = brokerage + STT + exchange fees + GST
        Market impact            = vol-adjusted price impact for large orders
        Net P&L                  = Gross P&L - slippage - explicit - market_impact

    WHY TCA IS IMPORTANT IN INTERVIEWS:
        "I implemented TCA to attribute P&L. Our strategy had a 0.15 Sharpe gross
        but after accounting for 0.06% round-trip costs, net Sharpe dropped to 0.08 —
        below our minimum threshold of 0.10. We increased signal threshold to trade
        less frequently but with higher conviction."

    Parameters
    ----------
    trades_df    : pd.DataFrame - Trade log (from load_trade_log())
                                  Must have columns: symbol, side, fill_price,
                                  intended_price, quantity, strategy, timestamp_utc
    market_data_df : pd.DataFrame (optional) - OHLCV data for market impact estimation

    Returns
    -------
    dict : TCA metrics including slippage breakdown, cost attribution, and net P&L.
    """
    if trades_df is None or trades_df.empty:
        return {'error': 'No trades to analyze.'}

    results = []

    # Match BUY-SELL pairs for round-trip analysis
    # Group by symbol and strategy, then pair consecutive buy/sell
    for symbol in trades_df['symbol'].unique():
        sym_trades = trades_df[trades_df['symbol'] == symbol].copy()
        sym_trades = sym_trades.sort_values('timestamp_utc')

        # Simple pairing: alternate BUY/SELL
        buys  = sym_trades[sym_trades['side'] == 'BUY'].reset_index(drop=True)
        sells = sym_trades[sym_trades['side'] == 'SELL'].reset_index(drop=True)

        n_pairs = min(len(buys), len(sells))
        for i in range(n_pairs):
            buy  = buys.iloc[i]
            sell = sells.iloc[i]
            qty  = min(int(buy.get('quantity', 1)), int(sell.get('quantity', 1)))

            # Intended vs Fill slippage
            buy_slippage  = float(buy.get('fill_price', 0)) - float(buy.get('intended_price', buy.get('fill_price', 0)))
            sell_slippage = float(sell.get('intended_price', sell.get('fill_price', 0))) - float(sell.get('fill_price', 0))
            total_slippage_pts = buy_slippage + sell_slippage
            total_slippage_inr = total_slippage_pts * qty

            # Gross P&L (fill price based)
            gross_pnl_pts = float(sell.get('fill_price', 0)) - float(buy.get('fill_price', 0))
            gross_pnl_inr = gross_pnl_pts * qty

            # Explicit transaction costs (round-trip)
            try:
                instrument = buy.get('instrument', 'equity')
                cost_info  = compute_round_trip_cost(
                    float(buy.get('fill_price', 0)),
                    float(sell.get('fill_price', 0)),
                    qty,
                    instrument,
                )
                explicit_cost_inr = cost_info['total_cost_inr']
            except Exception:
                explicit_cost_inr = abs(gross_pnl_inr) * 0.001  # 0.1% fallback

            # Market impact estimate (simplified: proportional to trade size × vol)
            market_impact_inr = _estimate_market_impact(
                price=float(buy.get('fill_price', 0)),
                quantity=qty,
                market_data_df=market_data_df,
                symbol=symbol,
            )

            net_pnl_inr = gross_pnl_inr - explicit_cost_inr - total_slippage_inr - market_impact_inr

            results.append({
                'symbol':              symbol,
                'entry_fill':          round(float(buy.get('fill_price', 0)), 2),
                'exit_fill':           round(float(sell.get('fill_price', 0)), 2),
                'quantity':            qty,
                'strategy':            buy.get('strategy', 'unknown'),
                'gross_pnl_pts':       round(gross_pnl_pts, 4),
                'gross_pnl_inr':       round(gross_pnl_inr, 2),
                'slippage_pts':        round(total_slippage_pts, 4),
                'slippage_inr':        round(total_slippage_inr, 2),
                'explicit_cost_inr':   round(explicit_cost_inr, 4),
                'market_impact_inr':   round(market_impact_inr, 4),
                'net_pnl_inr':         round(net_pnl_inr, 2),
                'cost_pct_of_gross':   round(
                    (explicit_cost_inr + total_slippage_inr) / abs(gross_pnl_inr) * 100
                    if gross_pnl_inr != 0 else 0.0, 2
                ),
            })

    if not results:
        return {
            'round_trips': [],
            'summary': {'n_pairs': 0, 'total_net_pnl': 0.0},
        }

    results_df     = pd.DataFrame(results)
    total_gross    = results_df['gross_pnl_inr'].sum()
    total_explicit = results_df['explicit_cost_inr'].sum()
    total_slippage = results_df['slippage_inr'].sum()
    total_impact   = results_df['market_impact_inr'].sum()
    total_net      = results_df['net_pnl_inr'].sum()

    summary = {
        'n_pairs':                 len(results),
        'total_gross_pnl_inr':    round(total_gross, 2),
        'total_explicit_cost_inr': round(total_explicit, 4),
        'total_slippage_inr':      round(total_slippage, 2),
        'total_market_impact_inr': round(total_impact, 4),
        'total_net_pnl_inr':       round(total_net, 2),
        'cost_drag_pct':           round((total_explicit + total_slippage) / abs(total_gross) * 100
                                          if total_gross != 0 else 0.0, 2),
        'attribution': {
            'signal_alpha':       f'₹{total_gross:.2f}',
            'minus_slippage':     f'-₹{total_slippage:.2f}',
            'minus_explicit':     f'-₹{total_explicit:.2f}',
            'minus_mkt_impact':   f'-₹{total_impact:.2f}',
            'equals_net_pnl':     f'₹{total_net:.2f}',
        },
    }

    return {
        'round_trips': results,
        'summary':     summary,
    }


def _estimate_market_impact(price: float, quantity: int,
                               market_data_df: pd.DataFrame = None,
                               symbol: str = '') -> float:
    """
    Estimate market impact using the Almgren-Chriss (2001) framework.

    The AC model separates impact into two economically distinct components:

        Temporary impact  h(v) = eta  * (Q / ADV)^alpha * price
            Cost of trading at participation rate v = Q/ADV.
            This is the bid-ask spread + short-term price pressure.
            It FULLY RECOVERS after the trade completes.

        Permanent impact  g(v) = gamma * (Q / ADV) * price
            Persistent price shift caused by information leakage.
            The market infers your intent from order flow and reprices.
            This does NOT recover.

        Total cost (single execution, no schedule decomposition):
            I = (temp_impact_per_unit + perm_impact_per_unit) * quantity

    Note on parameters:
        The full AC framework solves for an OPTIMAL LIQUIDATION SCHEDULE
        (how to slice a large order over time T to minimise expected cost +
        risk aversion * variance of cost). This function computes the
        lump-sum impact for a single market order — a valid simplification
        for small orders (Q << ADV) where schedule optimisation adds little.

        Calibrate eta and gamma from your own order book / fill data.
        Literature values for US equities (Almgren et al., 2005):
            eta   ~= 0.142  (temporary)
            gamma ~= 0.314  (permanent)
        Conservative NSE mid-cap estimates used below until calibration:
            eta   = 0.10
            gamma = 0.05

    ACADEMIC REFERENCE:
        Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions."
        Journal of Risk 3(2): 5-39.
        Almgren et al. (2005), "Direct Estimation of Equity Market Impact."
        Risk 18(7): 58-62.

    Parameters
    ----------
    price          : float         - Fill price per unit
    quantity       : int           - Order size (units / shares)
    market_data_df : pd.DataFrame  - OHLCV data to compute ADV (21-day)
    symbol         : str           - Symbol for filtering panel DataFrames

    Returns
    -------
    float : Total market impact cost in ₹. Returns 0.0 when ADV cannot be computed.
    """
    # Almgren-Chriss model parameters (calibrate from live fill data).
    # Until calibration data is available, use conservative NSE estimates.
    ETA   = 0.10   # Temporary impact coefficient
    GAMMA = 0.05   # Permanent impact coefficient
    ALPHA = 0.5    # Square-root law exponent (empirically robust; Almgren et al., 2005)

    if market_data_df is None or market_data_df.empty:
        # Cannot compute ADV without market data.
        # Return 0 rather than a fabricated percentage — unknown is better than wrong.
        return 0.0

    try:
        sym_data = (
            market_data_df[market_data_df['symbol'] == symbol]
            if 'symbol' in market_data_df.columns
            else market_data_df
        )
        if sym_data.empty:
            return 0.0

        adv = float(sym_data['volume'].tail(21).mean())
        if adv <= 0:
            return 0.0

        participation_rate = quantity / adv  # Q / ADV: fraction of daily volume traded

        # Temporary impact per unit: recovers fully after execution
        temp_impact_per_unit = ETA * (participation_rate ** ALPHA) * price

        # Permanent impact per unit: persistent repricing from information leakage
        perm_impact_per_unit = GAMMA * participation_rate * price

        total_impact_inr = (temp_impact_per_unit + perm_impact_per_unit) * quantity
        return round(total_impact_inr, 4)

    except Exception:
        return 0.0


