"""
quandao_public/config.py
=========================
Platform-wide configuration constants for Quandao.

HOW TO USE:
    from quandao_public.config import SIMULATED_CAPITAL_INR, RISK_FREE_RATE

DESIGN PRINCIPLE:
    Every number that a strategy or execution module needs (capital, rates, limits)
    lives here. Change one value here → affects ALL strategies simultaneously.
    This prevents the bug of hardcoding "100000" in 10 different files.
"""

# ── Simulated Paper Trading Capital ────────────────────────────────────────
# These are the starting balances for ALL dry-run simulations.
# No real money is ever used — DRY_RUN is always True.
SIMULATED_CAPITAL_INR = 100_000   # ₹1,00,000 — for NSE equity, options, NIFTY futures
SIMULATED_CAPITAL_USD = 1_000     # $1,000    — for Forex pairs (EURUSD, GBPUSD, etc.)

# ── Execution Mode ──────────────────────────────────────────────────────────
# ALWAYS True. Changing to False would route orders to the live Fyers API.
# Do NOT change this without understanding the live execution risk.
DRY_RUN = True

# ── Risk-Free Rate ──────────────────────────────────────────────────────────
# India risk-free rate = RBI repo rate (as of FY2025-26).
# Used by: Sharpe ratio denominator, Black-Scholes discounting, Kelly criterion.
RISK_FREE_RATE = 0.065   # 6.5% annualized

# ── NSE Market Session ──────────────────────────────────────────────────────
MARKET_OPEN_TIME  = "09:15"   # NSE opens at 9:15 AM IST
MARKET_CLOSE_TIME = "15:30"   # NSE closes at 3:30 PM IST
NSE_TIMEZONE      = "Asia/Kolkata"

# ── Risk Management Limits ──────────────────────────────────────────────────
# These caps prevent any single trade from blowing up the simulated account.

MAX_RISK_PER_TRADE_PCT = 0.02   # Risk at most 2% of capital per trade
                                 # On ₹1,00,000 → max risk = ₹2,000 per trade

MAX_PORTFOLIO_DELTA    = 0.25   # Options: aggregate portfolio delta must stay ≤ ±0.25
                                 # Prevents unhedged directional blow-up

MAX_DRAWDOWN_HALT      = 0.10   # Halt ALL new trades if simulated DD > 10% of capital
                                 # On ₹1,00,000 → halt at ₹10,000 cumulative loss

THETA_BURN_THRESHOLD   = 0.05   # Options: exit position if daily theta > 5% of premium paid
                                 # Prevents holding options as they decay to zero

VOL_SCALE_TARGET       = 0.15   # Target annualized volatility for position scaling
                                 # When market vol spikes above this, reduce position size

# ── NSE Instrument Lot Sizes ────────────────────────────────────────────────
NIFTY_LOT_SIZE     = 75    # NIFTY50 futures & options lot size (as of 2024 revision)
BANKNIFTY_LOT_SIZE = 30    # BANKNIFTY futures & options lot size
FINNIFTY_LOT_SIZE  = 40    # FINNIFTY options lot size

# ── NSE Transaction Cost Components (FY2025-26) ────────────────────────────
# Source: NSE circular + SEBI schedule of charges
# Used by cost_model.py to compute realistic per-trade cost.
BROKERAGE_PCT          = 0.0003    # 0.03% per leg (discount broker; capped at ₹20/order)
BROKERAGE_CAP_INR      = 20.0      # ₹20 maximum brokerage per order leg
STT_SELL_PCT           = 0.0001    # 0.01% STT on sell side (equity futures)
STT_OPTIONS_SELL_PCT   = 0.0625    # 6.25% on options premium (sell side only)
EXCHANGE_CHARGE_PCT    = 0.0000019 # 0.00019% NSE exchange transaction charge
SEBI_FEE_PCT           = 0.000001  # ₹10 per crore = 0.0001% (per leg)
STAMP_DUTY_BUY_PCT     = 0.00003   # 0.003% stamp duty on buy side
GST_RATE               = 0.18      # 18% GST on (brokerage + exchange charge + SEBI fee)

# ── File Paths ──────────────────────────────────────────────────────────────
# Relative to the project root (Quandao-main/).
PAPER_TRADE_LOG_PATH   = "quandao_private/results/live/paper_trades.json"
WALK_FORWARD_RESULTS   = "quandao_private/results/backtest/walk_forward_results.json"
OPTIONS_TRADE_LOG_PATH = "quandao_private/results/live/options_trades.json"

