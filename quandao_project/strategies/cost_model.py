"""
quandao_project/strategies/cost_model.py
==========================================
Realistic NSE transaction cost model — the difference between a toy backtest
and a production-grade one.

WHAT THIS MODULE DOES:
    Computes every cost component of an NSE equity/futures/options trade,
    exactly as a broker would charge it.

WHY THIS MATTERS (Interview Talking Point):
    Every naive backtester ignores costs. A strategy with 0.10% average
    profit per trade can be WIPED OUT by ~0.08–0.12% round-trip costs.
    This module makes that impossible to ignore.

NSE COST COMPONENTS (FY2025-26):
    1. Brokerage         — 0.03% per leg, capped at ₹20/order (discount broker)
    2. STT               — Securities Transaction Tax (sell-side only)
    3. Exchange charge   — 0.00019% (NSE transaction fee)
    4. SEBI fee          — ₹10 per crore turnover (0.0001%)
    5. Stamp duty        — 0.003% on buy-side only
    6. GST               — 18% on (brokerage + exchange charge + SEBI fee)

USAGE:
    from quandao_project.strategies.cost_model import compute_round_trip_cost

    cost = compute_round_trip_cost(entry_price=24500, exit_price=24550,
                                   quantity=75, instrument='futures')
    print(cost['total_cost_pts'])  # Cost in index points
    print(cost['total_cost_inr'])  # Cost in rupees
"""

from quandao_project.config import (
    BROKERAGE_PCT, BROKERAGE_CAP_INR,
    STT_SELL_PCT, STT_OPTIONS_SELL_PCT,
    EXCHANGE_CHARGE_PCT, SEBI_FEE_PCT,
    STAMP_DUTY_BUY_PCT, GST_RATE,
)


# ── Per-Leg Cost Computation ────────────────────────────────────────────────

def compute_leg_cost(price: float,
                     quantity: int,
                     side: str,
                     instrument: str = 'equity') -> dict:
    """
    Compute all cost components for a SINGLE order leg (one side of a trade).

    Parameters
    ----------
    price      : float  - Fill price (e.g. index level for futures, ₹ for equity)
    quantity   : int    - Number of shares/units (for futures: lot_size * n_lots)
    side       : str    - 'BUY' or 'SELL'
    instrument : str    - 'equity', 'futures', or 'options'
                          STT rules differ by instrument type.

    Returns
    -------
    dict with itemized costs (all in ₹) and total.
    """
    side = side.upper()
    instrument = instrument.lower()
    turnover = price * quantity  # Total trade value in ₹

    # 1. Brokerage: 0.03% of turnover, capped at ₹20 per leg
    brokerage = min(turnover * BROKERAGE_PCT, BROKERAGE_CAP_INR)

    # 2. STT — charged on the sell side only (per SEBI rules)
    stt = 0.0
    if side == 'SELL':
        if instrument == 'options':
            # For options: STT = 6.25% of PREMIUM value (not notional)
            # Since we pass the premium as `price`, this is correct.
            stt = turnover * STT_OPTIONS_SELL_PCT
        else:
            # For equity and futures: STT = 0.01% of sell turnover
            stt = turnover * STT_SELL_PCT

    # 3. Exchange transaction charge: 0.00019% of turnover (both sides)
    exchange_charge = turnover * EXCHANGE_CHARGE_PCT

    # 4. SEBI fee: ₹10 per crore = 0.000001 of turnover (both sides)
    sebi_fee = turnover * SEBI_FEE_PCT

    # 5. Stamp duty: 0.003% on BUY side only
    stamp_duty = turnover * STAMP_DUTY_BUY_PCT if side == 'BUY' else 0.0

    # 6. GST: 18% on (brokerage + exchange charge + SEBI fee)
    #    Stamp duty is NOT subject to GST.
    gst = (brokerage + exchange_charge + sebi_fee) * GST_RATE

    total_inr = brokerage + stt + exchange_charge + sebi_fee + stamp_duty + gst

    return {
        'turnover_inr':      round(turnover, 2),
        'brokerage_inr':     round(brokerage, 4),
        'stt_inr':           round(stt, 4),
        'exchange_charge_inr': round(exchange_charge, 4),
        'sebi_fee_inr':      round(sebi_fee, 6),
        'stamp_duty_inr':    round(stamp_duty, 4),
        'gst_inr':           round(gst, 4),
        'total_inr':         round(total_inr, 4),
    }


# ── Round-Trip Cost (Entry + Exit) ──────────────────────────────────────────

def compute_round_trip_cost(entry_price: float,
                             exit_price: float,
                             quantity: int,
                             instrument: str = 'equity') -> dict:
    """
    Compute total round-trip cost for a completed trade (entry buy + exit sell).

    INTERVIEW INSIGHT:
        A 50-point profit on NIFTY futures (lot=75) = ₹3,750 gross.
        Round-trip cost at current NSE rates ≈ ₹60–90.
        That's 1.6–2.4% of the trade's gross profit — not negligible.

    Parameters
    ----------
    entry_price : float  - Entry fill price
    exit_price  : float  - Exit fill price
    quantity    : int    - Total units (shares or futures contracts * lot_size)
    instrument  : str    - 'equity', 'futures', or 'options'

    Returns
    -------
    dict with:
        'entry_cost'         : dict (itemized entry leg costs)
        'exit_cost'          : dict (itemized exit leg costs)
        'total_cost_inr'     : float (total ₹ cost both legs)
        'total_cost_pts'     : float (cost expressed in price-per-unit points)
        'cost_as_pct_profit' : float (cost as % of the gross profit, if profitable)
        'breakeven_pts'      : float (minimum price move needed just to cover costs)
    """
    entry_cost = compute_leg_cost(entry_price, quantity, 'BUY', instrument)
    exit_cost  = compute_leg_cost(exit_price, quantity, 'SELL', instrument)

    total_cost_inr = entry_cost['total_inr'] + exit_cost['total_inr']

    # Express cost in per-unit price points (useful for strategy P&L comparisons)
    total_cost_pts = total_cost_inr / quantity if quantity > 0 else 0.0

    # How many points must price move just to break even on costs?
    breakeven_pts = total_cost_pts

    # Gross profit in ₹
    gross_pnl_inr = (exit_price - entry_price) * quantity
    cost_as_pct_profit = (
        (total_cost_inr / abs(gross_pnl_inr)) * 100
        if gross_pnl_inr != 0 else float('inf')
    )

    return {
        'entry_cost':          entry_cost,
        'exit_cost':           exit_cost,
        'total_cost_inr':      round(total_cost_inr, 4),
        'total_cost_pts':      round(total_cost_pts, 4),
        'breakeven_pts':       round(breakeven_pts, 4),
        'gross_pnl_inr':       round(gross_pnl_inr, 2),
        'net_pnl_inr':         round(gross_pnl_inr - total_cost_inr, 2),
        'cost_as_pct_profit':  round(cost_as_pct_profit, 2),
    }


def cost_adjusted_pnl(raw_pnl_pts: float,
                       entry_price: float,
                       quantity: int,
                       instrument: str = 'equity') -> float:
    """
    Quick helper: subtract round-trip cost from raw P&L (expressed in points).

    USAGE in strategy run() loops:
        raw_pnl = exit_price - entry_price  # Long trade
        net_pnl_pts = cost_adjusted_pnl(raw_pnl, entry_price, quantity)

    Parameters
    ----------
    raw_pnl_pts : float  - Gross P&L in price points (exit - entry for long)
    entry_price : float  - Entry price (used to compute turnover for costs)
    quantity    : int    - Position size in units
    instrument  : str    - 'equity', 'futures', or 'options'

    Returns
    -------
    float : Net P&L in points after deducting round-trip transaction costs.
    """
    exit_price = entry_price + raw_pnl_pts
    costs = compute_round_trip_cost(entry_price, exit_price, quantity, instrument)
    return round(raw_pnl_pts - costs['total_cost_pts'], 4)
