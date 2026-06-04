"""
quandao_project/strategies/position_sizing.py
===============================================
Position sizing engine — determines HOW MUCH to trade, not just WHEN.

WHY THIS MODULE EXISTS (Interview Talking Point):
    Most beginners focus on entry/exit signals. Professionals know that
    position sizing contributes MORE to long-term P&L than signal quality.
    The Kelly Criterion, proven mathematically, maximizes long-run wealth
    growth. Fixed fractional sizing is the safe institutional standard.

DESIGN:
    Pure functions only — no state, no side effects, easy to audit.

USAGE:
    from quandao_project.strategies.position_sizing import fixed_fractional, kelly_fraction

    # How many NIFTY futures contracts should we buy?
    n_lots = fixed_fractional(
        capital=100_000,         # ₹1,00,000 simulated capital
        risk_pct=0.02,           # Risk 2% per trade
        entry=24500,             # NIFTY entry level
        stop_loss=24450,         # Stop at 24450 → risk = 50 pts
        lot_size=75,             # NIFTY lot size
    )
"""

import numpy as np
from quandao_project.config import (
    SIMULATED_CAPITAL_INR,
    SIMULATED_CAPITAL_USD,
    MAX_RISK_PER_TRADE_PCT,
    NIFTY_LOT_SIZE,
)


# ── Kelly Criterion ─────────────────────────────────────────────────────────

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Full Kelly Criterion — optimal fraction of capital to risk per trade.

    FORMULA:
        f* = (p * b - q) / b
        where:
            p = win_rate
            q = 1 - p  (loss probability)
            b = avg_win / avg_loss  (win/loss ratio, also called "b" or "odds")

    MATHEMATICAL PROOF:
        Kelly maximizes the expected logarithm of wealth (geometric growth rate).
        Any fraction LARGER than Kelly guarantees eventual ruin (proven by Kelly, 1956).
        In practice: use HALF Kelly (f* / 2) to reduce variance.

    PRACTICAL LIMITS:
        - Kelly can return > 1.0 (leverage) for very high edge strategies.
          Cap at 0.25 (25% of capital) for safety.
        - Negative Kelly → expected negative edge → don't trade.

    Parameters
    ----------
    win_rate : float  - Historical win rate in [0, 1]
    avg_win  : float  - Average profit per winning trade (absolute, same unit as avg_loss)
    avg_loss : float  - Average loss per losing trade (absolute value, positive)

    Returns
    -------
    float : Fraction of capital to risk [0, 0.25 capped].
            Use half_kelly() for the practical version.
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0

    p = win_rate
    q = 1.0 - p
    b = avg_win / avg_loss  # Win/loss ratio

    full_kelly = (p * b - q) / b

    # Cap at 25% (institutional standard: never risk > 25% on a single bet)
    return round(max(0.0, min(full_kelly, 0.25)), 6)


def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Half Kelly — the standard institutional position sizing approach.

    WHY HALF KELLY:
        Full Kelly maximizes long-run growth but has ENORMOUS variance.
        Half Kelly gives ~75% of the growth rate with roughly 50% the variance.
        Most professional traders use Half Kelly or less.

    Returns
    -------
    float : Half of the full Kelly fraction.
    """
    return round(kelly_fraction(win_rate, avg_win, avg_loss) / 2.0, 6)


# ── Fixed Fractional Sizing ─────────────────────────────────────────────────

def fixed_fractional(capital: float,
                      risk_pct: float,
                      entry: float,
                      stop_loss: float,
                      lot_size: int = 1) -> int:
    """
    Fixed Fractional Position Sizing — the institutional standard.

    FORMULA:
        capital_at_risk = capital * risk_pct
        points_at_risk  = |entry - stop_loss|
        units           = capital_at_risk / points_at_risk
        n_lots          = floor(units / lot_size)

    EXAMPLE:
        Capital = ₹1,00,000 | Risk 2% = ₹2,000
        Entry = 24,500 | Stop Loss = 24,450 → Risk = 50 points
        Units = 2000 / 50 = 40 units → With lot_size=75: 0 lots (below 1 lot)

        This is REALISTIC — ₹1,00,000 capital is insufficient for NIFTY futures
        with tight stop losses. The model correctly returns 0 (don't trade).

    Parameters
    ----------
    capital   : float  - Available trading capital (₹ or $)
    risk_pct  : float  - Maximum fraction of capital to risk (e.g. 0.02 = 2%)
    entry     : float  - Entry price
    stop_loss : float  - Stop loss price
    lot_size  : int    - Minimum tradeable unit (1 for equity, 75 for NIFTY)

    Returns
    -------
    int : Number of complete lots to trade (0 if insufficient capital for even 1 lot).
    """
    if entry == stop_loss:
        return 0

    capital_at_risk = capital * risk_pct
    pts_at_risk = abs(entry - stop_loss)

    if pts_at_risk <= 0:
        return 0

    units = capital_at_risk / pts_at_risk
    n_lots = int(units // lot_size) if lot_size > 1 else int(units)

    return max(0, n_lots)


def fixed_fractional_shares(capital: float,
                              risk_pct: float,
                              entry: float,
                              stop_loss: float) -> int:
    """
    Fixed Fractional for equity shares (lot_size = 1).

    Same as fixed_fractional but returns individual share count.
    Used for NSE equity positions (e.g. RELIANCE, HDFCBANK).

    Returns
    -------
    int : Number of shares to buy.
    """
    return fixed_fractional(capital, risk_pct, entry, stop_loss, lot_size=1)


# ── Options-Specific Sizing ─────────────────────────────────────────────────

def delta_adjusted_size(capital: float,
                         target_portfolio_delta: float,
                         option_delta: float,
                         spot_price: float,
                         lot_size: int = NIFTY_LOT_SIZE,
                         max_premium_pct: float = 0.05) -> dict:
    """
    Delta-adjusted position sizing for options.

    WHY DELTA MATTERS FOR SIZING:
        Buying 1 lot of a 0.50-delta call = equivalent directional exposure
        as being long 37.5 units of spot (0.50 * 75 = 37.5).
        To control total portfolio delta, you size by target_delta / option_delta.

    FORMULA:
        n_lots = target_portfolio_delta / (option_delta * lot_size / spot_notional)
        → Simplified: n_lots such that (n_lots * lot_size * delta) / (capital/spot) = target

    PREMIUM CAP:
        Also cap at max_premium_pct of capital to prevent over-allocation to options.
        Options can go to zero — you should never put >5% of capital in premium.

    Parameters
    ----------
    capital               : float - Total simulated capital
    target_portfolio_delta: float - Desired net portfolio delta (e.g. 0.10 = 10% bullish)
    option_delta          : float - Delta of the chosen option (e.g. 0.40 for OTM call)
    spot_price            : float - Current spot price of underlying
    lot_size              : int   - Option lot size (75 for NIFTY)
    max_premium_pct       : float - Max % of capital allocatable to premium (safety cap)

    Returns
    -------
    dict with:
        'n_lots'                 : int   - Number of lots to trade
        'portfolio_delta'        : float - Resulting portfolio delta
        'max_premium_allocation' : float - Maximum ₹ to spend on premium
        'sizing_method'          : str   - 'delta_adjusted' or 'premium_capped'
    """
    if option_delta == 0 or spot_price == 0:
        return {'n_lots': 0, 'portfolio_delta': 0.0,
                'max_premium_allocation': 0.0, 'sizing_method': 'zero_delta'}

    # Delta-based sizing:
    # Each lot contributes (option_delta * lot_size) delta units
    # We want: n_lots * option_delta * lot_size = target_portfolio_delta * spot_equivalent
    # Simplified institutional rule: n_lots = target_delta / (option_delta)
    # (assumes 1 lot per delta unit as a starting point)
    delta_based_lots = int(abs(target_portfolio_delta) / abs(option_delta))

    # Premium cap: never spend more than max_premium_pct of capital on options premium
    max_premium_inr = capital * max_premium_pct

    return {
        'n_lots': max(0, delta_based_lots),
        'portfolio_delta': round(delta_based_lots * option_delta * lot_size, 4),
        'max_premium_allocation': round(max_premium_inr, 2),
        'sizing_method': 'delta_adjusted',
    }


# ── Risk-of-Ruin Calculator ─────────────────────────────────────────────────

def risk_of_ruin(win_rate: float, risk_reward: float,
                 risk_pct: float = MAX_RISK_PER_TRADE_PCT,
                 ruin_threshold: float = 0.50) -> float:
    """
    Probability of losing a specified fraction of capital (risk of ruin).

    FORMULA (Kelly-derived, simplified):
        RoR = ((1 - edge) / (1 + edge)) ^ (ruin_threshold / risk_pct)
        edge = p*b - q (Kelly's edge)
        b = risk_reward ratio

    WHY THIS MATTERS:
        A strategy with 50% win rate and 1:1 reward:risk has edge = 0.
        → Risk of ruin = 100% (eventual ruin is certain).
        A strategy with 55% win rate and 1:1 R:R has edge = 0.10.
        → At 2% risk per trade, risk of ruin ≈ 0.001% (essentially safe).

    Parameters
    ----------
    win_rate     : float - Historical win rate
    risk_reward  : float - Average win / average loss ratio
    risk_pct     : float - Fraction of capital risked per trade
    ruin_threshold: float - Define "ruin" as losing this fraction of capital

    Returns
    -------
    float : Probability of ruin in [0, 1]
    """
    p = win_rate
    q = 1.0 - p
    b = risk_reward

    # Edge = expected value of the bet
    edge = p * b - q

    if edge <= 0:
        return 1.0  # Negative edge → certain ruin eventually

    # Risk of ruin formula
    ror_ratio = (1.0 - edge) / (1.0 + edge)
    n_bets_to_ruin = ruin_threshold / risk_pct

    return round(float(ror_ratio ** n_bets_to_ruin), 8)
