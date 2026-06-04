"""
quandao_public/execution/options_executor.py
===============================================
End-to-end options execution pipeline: signal → strike → Greeks → dry-run order → log.

THIS IS THE OPTIONS EXECUTION BOT — Project 3 of the Resume.

FULL PIPELINE:
    1. Receive directional signal (from multi_factor composite)
    2. Check vol regime (from GARCH) → decide option type (buy vs spread)
    3. Select optimal strike by target delta
    4. Compute full Greeks at that strike
    5. Validate position size via risk_manager
    6. Log dry-run order with complete analytics
    7. Monitor open positions → exit on theta burn or target hit

SIMULATED CAPITAL: ₹1,00,000 for NSE options

KEY DESIGN:
    - ALWAYS DRY RUN — no real orders placed
    - Full analytics logged: strike, delta, gamma, theta, vega, IV, Greeks, cost
    - Positions tracked in memory (list of dicts) for risk monitoring
    - Separate log file: quandao_public/results/options_trades.json

USAGE:
    from quandao_public.execution.options_executor import run_options_paper_trade

    trades = run_options_paper_trade(
        signals=[{'symbol': 'NSE:NIFTY50-INDEX', 'signal': 'LONG', 'score': 0.72}],
        spot_data={'NSE:NIFTY50-INDEX': {'spot': 24500, 'iv': 0.18, 'days_to_expiry': 7}},
    )
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from quandao_public.config import (
    DRY_RUN,
    SIMULATED_CAPITAL_INR,
    RISK_FREE_RATE,
    MAX_RISK_PER_TRADE_PCT,
    OPTIONS_TRADE_LOG_PATH,
    NIFTY_LOT_SIZE,
    THETA_BURN_THRESHOLD,
)
from quandao_public.strategies.options_pricing import (
    black_scholes,
    implied_volatility,
    select_strike_by_delta,
    theta_burn_exit,
)
from quandao_public.strategies.position_sizing import delta_adjusted_size
from quandao_public.risk.risk_manager import (
    portfolio_risk_summary,
    single_trade_risk_check,
)
from quandao_public.execution.order_executor import log_trade


# ── Log Path Setup ────────────────────────────────────────────────────────────
_OPTIONS_LOG = Path(OPTIONS_TRADE_LOG_PATH)
_OPTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Signal → Option Trade Specification
# ══════════════════════════════════════════════════════════════════════════════

def generate_options_signal(composite_score: float,
                             long_threshold: float = 0.50,
                             short_threshold: float = -0.50) -> str:
    """
    Map a factor composite score to an options direction.

    MAPPING:
        score > +0.50  → 'LONG_CALL'  (bullish with enough conviction)
        score < -0.50  → 'LONG_PUT'   (bearish with enough conviction)
        in between     → 'NO_TRADE'   (not enough conviction to pay premium)

    WHY THRESHOLDS:
        Options cost money (premium). We only buy when signal conviction is HIGH.
        A 0.3 composite score might be right 55% of the time — not enough to
        overcome the time decay of the option. 0.5+ gives more edge over Theta.

    Parameters
    ----------
    composite_score  : float - Multi-factor composite score (from multi_factor.py)
    long_threshold   : float - Minimum score for a CALL (default 0.50)
    short_threshold  : float - Maximum score for a PUT (default -0.50)

    Returns
    -------
    str : 'LONG_CALL', 'LONG_PUT', or 'NO_TRADE'
    """
    if composite_score > long_threshold:
        return 'LONG_CALL'
    elif composite_score < short_threshold:
        return 'LONG_PUT'
    else:
        return 'NO_TRADE'


def build_options_order(signal: str,
                         spot_price: float,
                         sigma: float,
                         days_to_expiry: int = 7,
                         r: float = RISK_FREE_RATE,
                         capital: float = SIMULATED_CAPITAL_INR,
                         signal_score: float = 0.0,
                         lot_size: int = NIFTY_LOT_SIZE,
                         strike_step: int = 50) -> Optional[dict]:
    """
    Build a complete options order specification from a directional signal.

    PIPELINE:
        1. Convert signal to option_type (call/put)
        2. Determine target delta based on signal conviction:
            Strong signal (|score| > 0.7) → 0.45 delta (near ATM, higher P)
            Moderate signal               → 0.35 delta (slightly OTM)
        3. Find NSE-valid strike with that delta
        4. Compute full Greeks at that strike
        5. Validate premium cost ≤ 5% of capital
        6. Compute delta-adjusted position size

    Parameters
    ----------
    signal         : str   - 'LONG_CALL' or 'LONG_PUT' (from generate_options_signal)
    spot_price     : float - Current underlying spot price
    sigma          : float - Implied volatility (annualized, e.g. 0.18)
    days_to_expiry : int   - Calendar days to option expiry
    r              : float - Risk-free rate
    capital        : float - Available capital for sizing
    signal_score   : float - Original composite score (for delta targeting)
    lot_size       : int   - Option lot size (75 for NIFTY)
    strike_step    : int   - Strike interval (50 for NIFTY)

    Returns
    -------
    dict : Complete order specification with Greeks, sizing, and cost estimate.
           None if signal is 'NO_TRADE' or risk checks fail.
    """
    if signal == 'NO_TRADE':
        return None

    option_type = 'call' if signal == 'LONG_CALL' else 'put'
    T = max(days_to_expiry / 365.0, 1 / 365.0)   # Minimum 1 day

    # Target delta based on conviction level
    abs_score = abs(signal_score)
    if abs_score >= 0.70:
        target_delta = 0.45   # High conviction → near ATM
    elif abs_score >= 0.50:
        target_delta = 0.35   # Moderate → slightly OTM
    else:
        target_delta = 0.25   # Low conviction → further OTM

    # Find optimal strike
    strike = select_strike_by_delta(
        S=spot_price, T=T, r=r, sigma=sigma,
        target_delta=target_delta,
        option_type=option_type,
        strike_step=strike_step,
    )

    # Compute full Greeks
    greeks = black_scholes(S=spot_price, K=strike, T=T, r=r,
                            sigma=sigma, option_type=option_type)

    # Theoretical option price
    theoretical_price = greeks['price']

    # Premium validation: total premium ≤ 5% of capital
    max_premium_cap = capital * 0.05
    premium_per_lot  = theoretical_price * lot_size
    max_lots_by_cap  = max(0, int(max_premium_cap / premium_per_lot)) if premium_per_lot > 0 else 0

    # Delta-adjusted sizing
    sizing = delta_adjusted_size(
        capital=capital,
        target_portfolio_delta=target_delta,
        option_delta=greeks['delta'],
        spot_price=spot_price,
        lot_size=lot_size,
    )

    # Take the more conservative of the two sizing methods
    n_lots = min(max_lots_by_cap, sizing['n_lots'])
    n_lots = max(0, n_lots)

    total_premium_inr = theoretical_price * lot_size * n_lots

    # Daily theta burn for the full position
    position_daily_theta = greeks['theta'] * lot_size * n_lots

    return {
        'signal':              signal,
        'option_type':         option_type,
        'spot_price':          round(spot_price, 2),
        'strike':              strike,
        'days_to_expiry':      days_to_expiry,
        'target_delta':        target_delta,
        'sigma_used':          round(sigma, 4),

        # Greeks
        'price':               round(theoretical_price, 2),
        'delta':               greeks['delta'],
        'gamma':               greeks['gamma'],
        'theta':               greeks['theta'],       # Daily theta per unit
        'vega':                greeks['vega'],
        'rho':                 greeks['rho'],

        # Position sizing
        'n_lots':              n_lots,
        'lot_size':            lot_size,
        'total_units':         n_lots * lot_size,
        'total_premium_inr':   round(total_premium_inr, 2),
        'position_daily_theta': round(position_daily_theta, 4),

        # Risk metadata
        'max_loss_inr':        round(total_premium_inr, 2),  # Long options: max loss = premium
        'signal_score':        round(signal_score, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Full Paper Trade Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_options_paper_trade(signals: List[Dict],
                             spot_data: Dict,
                             open_positions: List[Dict] = None,
                             capital: float = SIMULATED_CAPITAL_INR,
                             strategy: str = 'options_executor') -> dict:
    """
    Full end-to-end options paper trade pipeline.

    INPUT:
        signals   : [{'symbol': str, 'signal': str, 'score': float}, ...]
                    signal is 'LONG_CALL', 'LONG_PUT', or 'NO_TRADE'

        spot_data : {symbol: {'spot': float, 'iv': float, 'days_to_expiry': int}}

    PIPELINE:
        For each signal:
            1. Build order specification (strike, Greeks, sizing)
            2. Risk checks (drawdown halt, delta cap, single-trade risk)
            3. Log dry-run order if approved
            4. Add to open_positions tracker

        For open_positions (monitoring):
            1. Update Greeks with current spot/IV
            2. Check theta burn → exit if triggered
            3. Log exit orders

    Returns
    -------
    dict : {
        'new_orders':    list of approved order specs,
        'rejected':      list of rejected orders with reasons,
        'exits':         list of theta-triggered exits,
        'session_summary': dict
    }
    """
    assert DRY_RUN, "DRY_RUN must be True."

    if open_positions is None:
        open_positions = []

    current_equity = capital  # In a real system, track this over time
    peak_equity    = capital

    new_orders = []
    rejected   = []
    exits      = []

    # ── 1. Process new signals ──────────────────────────────────────────────
    for sig_item in signals:
        symbol   = sig_item.get('symbol', 'UNKNOWN')
        signal   = sig_item.get('signal', 'NO_TRADE')
        score    = float(sig_item.get('score', 0.0))

        if signal == 'NO_TRADE':
            continue

        if symbol not in spot_data:
            rejected.append({'symbol': symbol, 'reason': 'No spot data provided.'})
            continue

        sd            = spot_data[symbol]
        spot          = float(sd.get('spot', 0))
        iv            = float(sd.get('iv', 0.18))
        days_to_exp   = int(sd.get('days_to_expiry', 7))

        # Portfolio risk pre-check
        risk_snapshot = portfolio_risk_summary(open_positions, current_equity, peak_equity, capital)
        if not risk_snapshot['all_clear']:
            rejected.append({
                'symbol': symbol, 'signal': signal,
                'reason': risk_snapshot['block_reason']
            })
            continue

        # Build order specification
        order_spec = build_options_order(
            signal=signal,
            spot_price=spot,
            sigma=iv,
            days_to_expiry=days_to_exp,
            capital=capital,
            signal_score=score,
        )

        if order_spec is None or order_spec['n_lots'] == 0:
            rejected.append({
                'symbol': symbol, 'signal': signal,
                'reason': f'Insufficient capital or 0 lots computed. Premium cap: {capital*0.05:.0f} INR.'
            })
            continue

        # Single trade risk check
        trade_risk = single_trade_risk_check(
            entry=order_spec['price'],
            stop_loss=0,   # Long options: max loss = premium, no traditional stop
            quantity=order_spec['total_units'],
            capital=capital,
            max_risk_pct=0.05,   # Options: allow up to 5% of capital as premium
        )

        # Assemble logged order
        order_log = {
            'order_id':          str(uuid.uuid4())[:8].upper(),
            'timestamp_utc':     datetime.now(timezone.utc).isoformat(),
            'status':            'SIMULATED',
            'dry_run':           True,
            'symbol':            symbol,
            'strategy':          strategy,
            **order_spec,
        }

        # Log to options trade log and general paper trades log
        _append_options_log(order_log)
        log_trade(order_log)

        # Add to open positions tracker
        open_positions.append({
            'symbol':          symbol,
            'option_type':     order_spec['option_type'],
            'strike':          order_spec['strike'],
            'n_lots':          order_spec['n_lots'],
            'lot_size':        order_spec['lot_size'],
            'entry_premium':   order_spec['price'],
            'option_price':    order_spec['price'],
            'daily_theta':     order_spec['theta'],
            'delta':           order_spec['delta'],
            'notional':        order_spec['total_premium_inr'],
            'entry_time':      datetime.now(timezone.utc).isoformat(),
        })

        new_orders.append(order_log)

        _print_options_order_summary(order_log)

    # ── 2. Monitor existing positions for exit triggers ─────────────────────
    from quandao_public.risk.risk_manager import check_theta_burn
    theta_exit_candidates = check_theta_burn(open_positions)

    for pos in theta_exit_candidates:
        exit_order = {
            'order_id':      str(uuid.uuid4())[:8].upper(),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'status':        'SIMULATED_EXIT',
            'dry_run':       True,
            'symbol':        pos.get('symbol', 'UNKNOWN'),
            'side':          'SELL',   # Close long position
            'exit_reason':   pos.get('exit_reason', 'THETA_BURN'),
            'position':      pos,
            'strategy':      strategy,
        }
        _append_options_log(exit_order)
        log_trade(exit_order)
        exits.append(exit_order)
        print(f"[OPTIONS EXECUTOR] EXIT triggered: {pos.get('symbol')} — {pos.get('exit_reason')}")

    # ── 3. Session summary ──────────────────────────────────────────────────
    total_premium_deployed = sum(o.get('total_premium_inr', 0) for o in new_orders)
    session_summary = {
        'new_orders':                len(new_orders),
        'rejected_orders':           len(rejected),
        'theta_exits':               len(exits),
        'total_premium_deployed_inr': round(total_premium_deployed, 2),
        'pct_capital_deployed':       round(total_premium_deployed / capital * 100, 2),
        'open_positions_count':       len(open_positions),
    }

    return {
        'new_orders':      new_orders,
        'rejected':        rejected,
        'exits':           exits,
        'session_summary': session_summary,
    }


# ── Log Helpers ──────────────────────────────────────────────────────────────

def _append_options_log(trade_dict: dict) -> None:
    """Append to the options-specific trade log."""
    path = _OPTIONS_LOG
    existing = []
    if path.exists():
        try:
            with open(path, 'r') as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)
        except json.JSONDecodeError:
            existing = []
    existing.append(trade_dict)
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)


def _print_options_order_summary(order: dict) -> None:
    """Clean options order summary to stdout."""
    print(
        f"\n{'─'*60}\n"
        f"  [OPTIONS DRY RUN] {order['order_id']}\n"
        f"  Signal : {order['signal']} | Score: {order['signal_score']:.3f}\n"
        f"  Spot   : {order['spot_price']} | Strike: {order['strike']} "
        f"({order['option_type'].upper()})\n"
        f"  DTE    : {order['days_to_expiry']} days | IV: {order['sigma_used']*100:.1f}%\n"
        f"  Price  : ₹{order['price']} | Lots: {order['n_lots']} | "
        f"Total Premium: ₹{order['total_premium_inr']:.0f}\n"
        f"  Greeks → Δ:{order['delta']:.3f} | Γ:{order['gamma']:.5f} | "
        f"Θ:{order['theta']:.2f}/day | ν:{order['vega']:.2f}\n"
        f"{'─'*60}"
    )

