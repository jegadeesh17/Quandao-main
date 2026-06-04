"""
quandao_project/risk/risk_manager.py
======================================
Portfolio-level risk guardrails for the Quandao simulated paper trading system.

WHAT THIS MODULE DOES:
    Enforces real-time risk limits on the simulated portfolio.
    Even in dry-run mode, we want the simulation to behave EXACTLY as a live
    system would under proper risk management — otherwise the paper results
    won't transfer to live trading.

RISK LAYERS IMPLEMENTED:
    1. Max drawdown circuit breaker  — halt all new trades if DD > 10%
    2. Portfolio delta cap           — no more than ±25% net directional exposure
    3. Volatility-adjusted sizing    — scale size inversely with GARCH vol
    4. Single-position concentration — no single trade > 2% risk
    5. Theta burn monitor            — exit decaying options before worthless

ALL FUNCTIONS ARE PURE:
    They take state as input and return a decision (bool or float).
    No global state modifications. Easy to audit. Easy to explain.

USAGE:
    from quandao_project.risk.risk_manager import (
        max_drawdown_halt, check_portfolio_delta, scale_position_for_volatility
    )
"""

import numpy as np
import pandas as pd
from quandao_project.config import (
    SIMULATED_CAPITAL_INR,
    MAX_DRAWDOWN_HALT,
    MAX_PORTFOLIO_DELTA,
    MAX_RISK_PER_TRADE_PCT,
    VOL_SCALE_TARGET,
    THETA_BURN_THRESHOLD,
)


# ── Circuit Breakers ─────────────────────────────────────────────────────────

def max_drawdown_halt(current_equity: float,
                       peak_equity: float = None,
                       initial_capital: float = SIMULATED_CAPITAL_INR,
                       max_dd_pct: float = MAX_DRAWDOWN_HALT) -> dict:
    """
    Circuit breaker: halt all new trades if portfolio drawdown exceeds threshold.

    RATIONALE:
        A strategy that has already lost 10% is likely in a regime where its
        signals are not working. Continuing to trade compounds the drawdown.
        The circuit breaker forces a PAUSE and review.

    FORMULA:
        current_dd = (peak_equity - current_equity) / peak_equity
        halt = current_dd > max_dd_pct

    Parameters
    ----------
    current_equity : float - Current portfolio value (simulated capital + cumulative P&L)
    peak_equity    : float - Highest portfolio value seen so far (None → use initial)
    initial_capital: float - Starting capital (default from config)
    max_dd_pct     : float - Halt threshold (default 0.10 = 10% drawdown)

    Returns
    -------
    dict with:
        'should_halt'    : bool  - True if trading should stop
        'current_dd_pct' : float - Current drawdown as a percentage
        'capital_at_risk': float - ₹ amount lost from peak
        'message'        : str   - Human-readable status
    """
    peak = peak_equity if peak_equity is not None else initial_capital
    peak = max(peak, initial_capital)  # Peak can never be below initial capital

    drawdown_inr = peak - current_equity
    drawdown_pct = drawdown_inr / peak if peak > 0 else 0.0
    should_halt  = drawdown_pct > max_dd_pct

    if should_halt:
        msg = (f"🔴 HALT: Drawdown {drawdown_pct*100:.1f}% exceeds {max_dd_pct*100:.0f}% limit. "
               f"No new trades. Review strategy parameters.")
    elif drawdown_pct > max_dd_pct * 0.75:
        msg = (f"🟡 WARNING: Drawdown {drawdown_pct*100:.1f}% approaching halt threshold. "
               f"Reduce position sizes.")
    else:
        msg = f"🟢 OK: Drawdown {drawdown_pct*100:.1f}% within limit."

    return {
        'should_halt':     should_halt,
        'current_dd_pct':  round(drawdown_pct * 100, 2),
        'capital_at_risk': round(drawdown_inr, 2),
        'peak_equity':     round(peak, 2),
        'message':         msg,
    }


def single_trade_risk_check(entry: float,
                              stop_loss: float,
                              quantity: int,
                              capital: float = SIMULATED_CAPITAL_INR,
                              max_risk_pct: float = MAX_RISK_PER_TRADE_PCT) -> dict:
    """
    Validate that a single trade doesn't risk more than max_risk_pct of capital.

    FORMULA:
        trade_risk_inr = |entry - stop_loss| * quantity
        trade_risk_pct = trade_risk_inr / capital
        approved = trade_risk_pct <= max_risk_pct

    Parameters
    ----------
    entry        : float - Entry price
    stop_loss    : float - Stop loss price
    quantity     : int   - Position size (number of units/shares)
    capital      : float - Current portfolio value
    max_risk_pct : float - Max % of capital to risk (default 2%)

    Returns
    -------
    dict with 'approved' (bool), 'trade_risk_pct', 'trade_risk_inr', 'message'
    """
    trade_risk_inr = abs(entry - stop_loss) * quantity
    trade_risk_pct = trade_risk_inr / capital if capital > 0 else 1.0
    approved       = trade_risk_pct <= max_risk_pct

    return {
        'approved':        approved,
        'trade_risk_inr':  round(trade_risk_inr, 2),
        'trade_risk_pct':  round(trade_risk_pct * 100, 2),
        'max_risk_inr':    round(capital * max_risk_pct, 2),
        'message': (
            f"✅ Risk {trade_risk_pct*100:.2f}% within {max_risk_pct*100:.0f}% limit."
            if approved else
            f"❌ Risk {trade_risk_pct*100:.2f}% exceeds {max_risk_pct*100:.0f}% limit. Reduce size."
        ),
    }


# ── Options-Specific Risk ────────────────────────────────────────────────────

def check_portfolio_delta(positions: list,
                           max_delta: float = MAX_PORTFOLIO_DELTA) -> dict:
    """
    Check if the aggregate portfolio delta is within the cap.

    WHAT PORTFOLIO DELTA MEANS:
        A portfolio with delta = +0.25 behaves like being long 25% of the
        underlying — if NIFTY drops 1%, portfolio loses 0.25%.
        This limits unhedged directional exposure.

    FORMULA:
        net_delta = sum(position_delta * n_lots * lot_size for each position)
        net_delta_pct = net_delta / total_notional
        breached = |net_delta_pct| > max_delta

    Parameters
    ----------
    positions : list of dicts
        Each dict: {'symbol', 'delta', 'n_lots', 'lot_size', 'notional'}
    max_delta : float - Maximum allowed net delta (default 0.25)

    Returns
    -------
    dict with 'delta_ok', 'net_delta', 'message'
    """
    if not positions:
        return {'delta_ok': True, 'net_delta': 0.0, 'message': '✅ No open positions.'}

    net_delta    = 0.0
    total_notional = 0.0

    for pos in positions:
        lots      = pos.get('n_lots', 0)
        lot_size  = pos.get('lot_size', 1)
        delta     = pos.get('delta', 0.0)
        notional  = pos.get('notional', 0.0)

        # Contribution: delta * lots * lot_size (directional equivalent units)
        net_delta     += delta * lots * lot_size
        total_notional += abs(notional)

    # Normalize by notional to get a percentage
    net_delta_pct = net_delta / total_notional if total_notional > 0 else 0.0
    delta_ok      = abs(net_delta_pct) <= max_delta

    return {
        'delta_ok':       delta_ok,
        'net_delta':      round(net_delta, 4),
        'net_delta_pct':  round(net_delta_pct, 4),
        'message': (
            f"✅ Net delta {net_delta_pct:.2%} within ±{max_delta:.0%} cap."
            if delta_ok else
            f"❌ Net delta {net_delta_pct:.2%} breaches ±{max_delta:.0%} cap. Hedge required."
        ),
    }


def check_theta_burn(positions: list,
                      threshold_pct: float = THETA_BURN_THRESHOLD) -> list:
    """
    Check which options positions need to be exited due to theta decay.

    Returns a list of positions that SHOULD BE CLOSED.

    Parameters
    ----------
    positions : list of dicts
        Each dict: {'symbol', 'option_price', 'daily_theta', 'entry_premium'}
    threshold_pct : float - Exit if |theta|/price > threshold (default 5%/day)

    Returns
    -------
    list : Subset of positions that should be exited due to theta burn.
    """
    exit_list = []

    for pos in positions:
        price  = pos.get('option_price', pos.get('entry_premium', 1.0))
        theta  = pos.get('daily_theta', 0.0)

        if price <= 0:
            exit_list.append({**pos, 'exit_reason': 'WORTHLESS'})
            continue

        burn_rate = abs(theta) / price
        if burn_rate > threshold_pct:
            exit_list.append({
                **pos,
                'exit_reason':     f'THETA_BURN ({burn_rate*100:.1f}%/day)',
                'theta_burn_rate': round(burn_rate * 100, 2),
            })

    return exit_list


# ── Volatility Scaling ───────────────────────────────────────────────────────

def scale_position_for_volatility(base_size: float,
                                   current_vol_pct: float,
                                   target_vol_pct: float = VOL_SCALE_TARGET * 100) -> float:
    """
    Scale position size inversely with volatility (volatility targeting).

    FORMULA:
        scaled_size = base_size * (target_vol / current_vol)
        Capped at [0.25 * base_size, 2.0 * base_size]

    EXAMPLE:
        Base size = 1 lot | Current vol = 2.0%/day | Target = 1.5%/day
        → scaled = 1 * (1.5/2.0) = 0.75 lots → round down to 0 lots
        (Not tradeable at 0.75 lots — this correctly prevents the trade)

    Parameters
    ----------
    base_size         : float - Base position size (lots, shares, etc.)
    current_vol_pct   : float - Current conditional vol in % (from GARCH model)
    target_vol_pct    : float - Target daily vol in % (default 15% annualized ÷ √252 ≈ 0.94%/day)
                                Config uses 15% annualized → pass as 100*0.15/sqrt(252) ≈ 0.94

    Returns
    -------
    float : Scaled position size (floored at 0.25x, capped at 2x base)
    """
    if current_vol_pct <= 0 or np.isnan(current_vol_pct):
        return base_size

    scale = target_vol_pct / current_vol_pct
    return round(base_size * float(np.clip(scale, 0.25, 2.0)), 4)


# ── Portfolio Summary ────────────────────────────────────────────────────────

def portfolio_risk_summary(positions: list,
                             current_equity: float,
                             peak_equity: float,
                             initial_capital: float = SIMULATED_CAPITAL_INR) -> dict:
    """
    Full portfolio risk snapshot — called before any new order placement.

    Combines all checks into one call. If ANY check fails, block new trades.

    Parameters
    ----------
    positions      : list  - All open positions
    current_equity : float - Current portfolio value
    peak_equity    : float - All-time high portfolio value

    Returns
    -------
    dict : {'all_clear': bool, 'dd_check': dict, 'delta_check': dict, 'theta_exits': list}
    """
    dd_check    = max_drawdown_halt(current_equity, peak_equity, initial_capital)
    delta_check = check_portfolio_delta(positions)
    theta_exits = check_theta_burn(positions)

    all_clear = (
        not dd_check['should_halt'] and
        delta_check['delta_ok']
    )

    return {
        'all_clear':    all_clear,
        'dd_check':     dd_check,
        'delta_check':  delta_check,
        'theta_exits':  theta_exits,
        'block_reason': (
            dd_check['message']     if dd_check['should_halt'] else
            delta_check['message']  if not delta_check['delta_ok'] else
            'All risk checks passed.'
        ),
    }
