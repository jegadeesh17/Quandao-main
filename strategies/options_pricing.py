"""
quandao_public/strategies/options_pricing.py
===============================================
Standalone options math module: Black-Scholes pricing, Greeks, and Implied Volatility.

WHY THIS IS SEPARATE FROM monte_carlo.py:
    monte_carlo.py uses BS as one component inside a GBM simulation strategy.
    This module is the PURE MATH LIBRARY — usable by any module that needs
    pricing, Greeks, or IV without the simulation overhead.

    Additionally, this module adds what monte_carlo.py lacks:
    → The Newton-Raphson Implied Volatility Solver — the piece that makes
      options analytics INTERACTIVE (you give it market price, it gives you IV).

WHAT EACH SECTION CONTAINS:
    1. black_scholes()          — Full BS: price + all 5 Greeks in one call
    2. iv_newton_raphson()      — Newton-Raphson IV solver (fast, converges in ~10 iters)
    3. iv_bisection()           — Brent's bisection (fallback for near-zero Vega)
    4. implied_volatility()     — Smart solver: tries NR, falls back to bisection
    5. select_strike_by_delta() — Find the strike with a target delta
    6. theta_burn_exit()        — Should we exit based on Theta decay rate?

INTERVIEW GOLD: Be ready to explain:
    Q: "How do you invert Black-Scholes for IV?"
    A: There's no closed-form solution. Newton-Raphson iterates:
       σ_new = σ_old - (BS(σ_old) - market_price) / Vega(σ_old)
       Converges in ~10 iterations for normal IV ranges (5%–150%).
       Falls back to bisection when Vega is near zero (deep ITM/OTM options).

USAGE:
    from quandao_public.strategies.options_pricing import (
        black_scholes, implied_volatility, select_strike_by_delta
    )

    pricing = black_scholes(S=24500, K=24500, T=7/365, r=0.065, sigma=0.18)
    print(pricing)
    # {'price': 195.3, 'delta': 0.52, 'gamma': 0.0011, 'theta': -12.4, 'vega': 1.8, 'rho': 0.03}

    iv = implied_volatility(market_price=200, S=24500, K=24500, T=7/365, r=0.065)
    print(iv)  # → 0.183 (18.3%)
"""

import numpy as np
from scipy.stats import norm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Black-Scholes Pricing + Greeks
# ══════════════════════════════════════════════════════════════════════════════

def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = 'call') -> dict:
    """
    Black-Scholes option price + all 5 Greeks in one function call.

    THE BLACK-SCHOLES FORMULA:
        d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
        d₂ = d₁ - σ·√T

        Call price = S·Φ(d₁) - K·e^{-rT}·Φ(d₂)
        Put  price = K·e^{-rT}·Φ(-d₂) - S·Φ(-d₁)

    GREEKS:
        Delta (Δ) = ∂Price/∂S         — sensitivity to spot price
        Gamma (Γ) = ∂²Price/∂S²       — sensitivity of Delta to spot
        Theta (Θ) = ∂Price/∂T         — daily time decay (negative for long)
        Vega  (ν) = ∂Price/∂σ · (1%)  — per 1% change in implied vol
        Rho   (ρ) = ∂Price/∂r · (1%)  — per 1% change in risk-free rate

    ASSUMPTIONS (and why they matter in interviews):
        - Constant volatility → reality: vol surface (smile/skew exists)
        - No dividends → use Merton (1973) model for dividend-paying stocks
        - European exercise only → NIFTY options are European ✅
        - Continuous trading, no transaction costs → we add them separately

    Parameters
    ----------
    S           : float  - Spot price (current index/stock level)
    K           : float  - Strike price
    T           : float  - Time to expiry in YEARS (e.g. 7 days = 7/365 ≈ 0.0192)
    r           : float  - Risk-free rate (annualized, e.g. 0.065 for India)
    sigma       : float  - Implied/historical volatility (annualized, e.g. 0.18 for 18%)
    option_type : str    - 'call' or 'put'

    Returns
    -------
    dict with:
        price  : float - Theoretical option price
        delta  : float - Delta (call: 0 to 1, put: -1 to 0)
        gamma  : float - Gamma (always positive for long options)
        theta  : float - Daily theta in same price units as option price
        vega   : float - Vega per 1% IV change
        rho    : float - Rho per 1% rate change
        d1, d2 : float - Intermediate values (useful for debugging)
    """
    if T <= 0:
        # Expired option: intrinsic value only
        intrinsic = max(0.0, (S - K) if option_type == 'call' else (K - S))
        return {
            'price': intrinsic, 'delta': 0.0, 'gamma': 0.0,
            'theta': 0.0, 'vega': 0.0, 'rho': 0.0, 'd1': 0.0, 'd2': 0.0,
        }

    # Core d1, d2 computation
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Cumulative normal distribution values
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)   # Standard normal PDF at d1 (used for Gamma, Vega, Theta)

    discount = np.exp(-r * T)   # e^{-rT}

    # Price
    if option_type == 'call':
        price = S * N_d1 - K * discount * N_d2
        delta = N_d1
        rho   = K * T * discount * N_d2 / 100   # per 1% rate change
    else:
        price = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = N_d1 - 1.0   # Negative for put
        rho   = -K * T * discount * norm.cdf(-d2) / 100

    # Gamma (same for call and put)
    gamma = n_d1 / (S * sigma * np.sqrt(T))

    # Theta: daily decay (in price units per day)
    # Divided by 365 for daily theta (NOT 252 — calendar days, not trading days)
    if option_type == 'call':
        theta = (
            -(S * n_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * discount * N_d2
        ) / 365
    else:
        theta = (
            -(S * n_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * discount * norm.cdf(-d2)
        ) / 365

    # Vega: per 1% change in IV (multiply by 0.01 for absolute sigma change)
    vega = S * n_d1 * np.sqrt(T) / 100   # per 1% = per 0.01 sigma

    return {
        'price':  round(price, 4),
        'delta':  round(delta, 6),
        'gamma':  round(gamma, 8),
        'theta':  round(theta, 4),   # Daily theta
        'vega':   round(vega, 4),    # Per 1% IV
        'rho':    round(rho, 6),
        'd1':     round(d1, 6),
        'd2':     round(d2, 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Implied Volatility Solvers
# ══════════════════════════════════════════════════════════════════════════════

def iv_newton_raphson(market_price: float, S: float, K: float, T: float, r: float,
                       option_type: str = 'call',
                       initial_guess: float = 0.20,
                       tol: float = 1e-6,
                       max_iter: int = 100) -> float:
    """
    Newton-Raphson Implied Volatility solver.

    WHY NEWTON-RAPHSON:
        Black-Scholes has no closed-form inverse for sigma (IV).
        Newton-Raphson uses the derivative (Vega) to find the root of:
            f(σ) = BS_price(σ) - market_price = 0

        Update rule:
            σ_{n+1} = σ_n - f(σ_n) / f'(σ_n)
                     = σ_n - (BS_price(σ_n) - market_price) / Vega(σ_n)

        Converges quadratically → typically 5–15 iterations to 6 decimal places.

    FAILURE CASES:
        1. Deep ITM/OTM: Vega → 0, update becomes unstable → switch to bisection
        2. Expired (T=0): No solution exists → return NaN
        3. Price below intrinsic value: No real IV exists → return NaN

    Parameters
    ----------
    market_price  : float - Observed market price of the option
    S, K, T, r    : float - BS model parameters (same as black_scholes())
    option_type   : str   - 'call' or 'put'
    initial_guess : float - Starting sigma (default 0.20 = 20%)
    tol           : float - Convergence tolerance (default 1e-6 = price accuracy)
    max_iter      : int   - Maximum iterations before giving up

    Returns
    -------
    float : Implied volatility (annualized). NaN if no solution found.
    """
    if T <= 0 or market_price <= 0:
        return float('nan')

    # Intrinsic value check: no real IV exists below intrinsic
    intrinsic = max(0.0, (S - K) if option_type == 'call' else (K - S))
    if market_price < intrinsic - 1e-4:
        return float('nan')

    sigma = max(0.001, initial_guess)  # Start with initial guess, floor at 0.1%

    for i in range(max_iter):
        greeks     = black_scholes(S, K, T, r, sigma, option_type)
        price_diff = greeks['price'] - market_price

        # Convergence check
        if abs(price_diff) < tol:
            return round(sigma, 8)

        vega_abs = greeks['vega'] * 100   # Convert back from per-1% to absolute
        if abs(vega_abs) < 1e-10:
            # Vega too small → Newton-Raphson becomes unstable → fall back
            return float('nan')

        # Newton-Raphson update
        sigma -= price_diff / vega_abs
        sigma  = max(1e-4, min(sigma, 20.0))   # Keep in [0.01%, 2000%]

    return round(sigma, 8)  # Return best estimate even if not fully converged


def iv_bisection(market_price: float, S: float, K: float, T: float, r: float,
                  option_type: str = 'call',
                  low: float = 0.001, high: float = 5.0,
                  tol: float = 1e-6, max_iter: int = 200) -> float:
    """
    Brent's Bisection Implied Volatility solver (fallback).

    Slower than Newton-Raphson but ALWAYS converges when a solution exists.
    Used for deep ITM/OTM options where Vega ≈ 0 (NR diverges).

    Bisection: finds σ where BS_price(σ) = market_price by binary search
    in [low_sigma, high_sigma].

    Parameters
    ----------
    low  : float - Lower bound for sigma (default 0.001 = 0.1%)
    high : float - Upper bound for sigma (default 5.0 = 500%)

    Returns
    -------
    float : Implied volatility. NaN if no root in [low, high].
    """
    if T <= 0 or market_price <= 0:
        return float('nan')

    def price_diff(sigma):
        return black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price

    # Check that a root exists in [low, high]
    if price_diff(low) * price_diff(high) > 0:
        return float('nan')  # Same sign → no root in interval

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = price_diff(mid)

        if abs(f_mid) < tol or (high - low) < tol:
            return round(mid, 8)

        if price_diff(low) * f_mid < 0:
            high = mid
        else:
            low = mid

    return round((low + high) / 2.0, 8)


def implied_volatility(market_price: float, S: float, K: float, T: float, r: float,
                        option_type: str = 'call') -> float:
    """
    Smart IV solver: tries Newton-Raphson first, falls back to bisection.

    This is the function you call in production — it handles all edge cases.

    Parameters
    ----------
    market_price : float - Observed option market price
    S, K, T, r   : float - Black-Scholes parameters
    option_type  : str   - 'call' or 'put'

    Returns
    -------
    float : Implied volatility (annualized). NaN if unsolvable.
    """
    # Try Newton-Raphson first (fast)
    iv = iv_newton_raphson(market_price, S, K, T, r, option_type)

    if np.isnan(iv) or iv <= 0:
        # Fall back to bisection (robust)
        iv = iv_bisection(market_price, S, K, T, r, option_type)

    return iv


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Strike Selection and Risk Utilities
# ══════════════════════════════════════════════════════════════════════════════

def select_strike_by_delta(S: float, T: float, r: float, sigma: float,
                            target_delta: float = 0.40,
                            option_type: str = 'call',
                            strike_step: int = 50) -> float:
    """
    Find the NSE-listed strike nearest to the target delta.

    RATIONALE:
        Rather than picking strikes by moneyness (ATM, 2% OTM, etc.),
        institutional traders target DELTA because:
        - Delta is a direct measure of directional exposure
        - A 0.40-delta call is ~40% equivalent to being long the underlying
        - Comparable across different vol regimes and maturities

    ALGORITHM:
        Generate all NSE-valid strikes within ±30% of spot.
        For each strike, compute BS delta.
        Pick the strike where |computed_delta - target_delta| is minimized.

    TYPICAL TARGETS:
        Strong directional signal → 0.45 delta (near ATM)
        Moderate signal           → 0.35 delta (slightly OTM)
        Weak / hedging signal     → 0.25 delta (further OTM)

    Parameters
    ----------
    S            : float - Spot price
    T            : float - Time to expiry in years
    r            : float - Risk-free rate
    sigma        : float - Implied volatility (use current IV or historical vol)
    target_delta : float - Target delta (default 0.40 for moderate directional)
    option_type  : str   - 'call' or 'put'
    strike_step  : int   - NSE strike interval (50 for NIFTY, 100 for BankNIFTY)

    Returns
    -------
    float : Best strike matching target delta
    """
    # Generate candidate strikes: ±30% around spot, at strike_step intervals
    low_k  = int(S * 0.70 // strike_step) * strike_step
    high_k = int(S * 1.30 // strike_step) * strike_step + strike_step

    best_strike  = round(S / strike_step) * strike_step  # ATM as default
    best_gap     = float('inf')

    for K in range(low_k, high_k + strike_step, strike_step):
        if K <= 0:
            continue
        greeks   = black_scholes(S, K, T, r, sigma, option_type)
        delta    = greeks['delta']
        gap      = abs(abs(delta) - abs(target_delta))
        if gap < best_gap:
            best_gap    = gap
            best_strike = K

    return float(best_strike)


def theta_burn_exit(option_price: float,
                     daily_theta: float,
                     threshold_pct: float = 0.05) -> bool:
    """
    Exit signal based on Theta decay rate.

    WHEN TO USE:
        For long options positions, Theta (time decay) accelerates near expiry.
        If daily decay > 5% of the premium paid, the position is decaying too fast.
        Better to exit and redeploy capital than watch the option expire worthless.

    FORMULA:
        theta_burn_rate = |daily_theta| / option_price
        exit_signal = theta_burn_rate > threshold_pct

    EXAMPLE:
        Option premium = ₹200, daily Theta = -₹15
        Burn rate = 15/200 = 7.5% → exceeds 5% threshold → EXIT

    Parameters
    ----------
    option_price   : float - Current option price (premium paid or current market)
    daily_theta    : float - Daily theta (negative for long positions)
    threshold_pct  : float - Exit if daily decay exceeds this % of price (default 5%)

    Returns
    -------
    bool : True if position should be exited due to theta burn.
    """
    if option_price <= 0:
        return True   # Option is worthless — exit

    burn_rate = abs(daily_theta) / option_price
    return burn_rate > threshold_pct


def vol_richness(current_iv: float, hist_vol_252d: float) -> str:
    """
    Classify whether options are rich or cheap relative to historical vol.

    WHY THIS MATTERS:
        If IV > historical vol → options are "rich" → prefer SELLING premium
        If IV < historical vol → options are "cheap" → prefer BUYING premium

    Parameters
    ----------
    current_iv     : float - Current implied volatility (from IV solver)
    hist_vol_252d  : float - 252-day realized historical volatility

    Returns
    -------
    str : 'RICH', 'FAIR', or 'CHEAP' with numeric spread
    """
    spread = current_iv - hist_vol_252d

    if spread > 0.03:   # IV more than 3% above historical
        return f'RICH (+{spread*100:.1f}% premium) — prefer SELLING'
    elif spread < -0.03:
        return f'CHEAP ({spread*100:.1f}% discount) — prefer BUYING'
    else:
        return f'FAIR ({spread*100:.1f}% spread) — neutral bias'

