"""
quandao_project/data/factor_engine.py
========================================
Vectorized alpha factor library for Nifty 50 position / swing trading.

Design Contract
---------------
* Pure transformations: DataFrame (daily OHLCV) -> pd.Series / pd.DataFrame.
* No DB calls, no API calls, no file I/O.
* NaN-safe: all rolling operations use min_periods to avoid silently dropping rows.
* @njit used only for heavy element-wise loops that cannot be expressed with
  Pandas rolling API (e.g. Amihud, where per-element conditional logic matters).

Factor Catalogue
----------------
ORIGINAL (from research phase)
  compute_amihud            -- Amihud Illiquidity Ratio (21-day)
  compute_momentum          -- 12-1 Month Cross-Sectional Momentum
  compute_low_volatility    -- Quarterly Low-Vol (63-day std, sign-flipped)

MEDIUM-TERM ALPHA EXPANSION (swing / position, 30d – 6m horizon)
  compute_intermediate_momentum          -- 6-Minus-1 Month Momentum (126d skip-21)
  compute_quarterly_low_volatility       -- Quarterly Low-Vol Anomaly (63-day std, –1x)
  compute_sharpe_adjusted_trend          -- Sharpe-Adjusted 90-Day Trend Strength
  compute_sma100_pullback                -- % Distance from 100-Day SMA
  compute_vpt_accumulation               -- 63-Day Institutional Volume-Price Trend
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


# ===========================================================================
# Numba kernels (computationally heavy rolling loops)
# ===========================================================================

@njit
def _calculate_amihud_illiquidity(returns: np.ndarray,
                                   volumes: np.ndarray,
                                   window: int) -> np.ndarray:
    """Numba-accelerated Amihud ILLIQ = mean(|R| / DollarVolume) over window."""
    n = len(returns)
    out = np.full(n, np.nan)
    for i in range(window, n):
        vol_window = volumes[i - window:i]
        ret_window = returns[i - window:i]
        valid = vol_window > 0
        if np.sum(valid) > 0:
            illiq = np.abs(ret_window[valid]) / vol_window[valid]
            out[i] = np.mean(illiq)
    return out


# ===========================================================================
# Original Factors
# ===========================================================================

def compute_amihud(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Amihud Illiquidity Ratio.

    ILLIQ_t = (1/D) * Σ |R_d| / DollarVolume_d

    Parameters
    ----------
    df     : Daily OHLCV DataFrame with 'close' and 'volume' columns.
    window : Rolling window in trading days (default 21 ~ 1 month).

    Returns
    -------
    pd.Series named 'amihud_illiq'. Higher values = more illiquid.
    """
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("DataFrame must contain 'close' and 'volume' columns.")

    returns = df["close"].pct_change().fillna(0.0).values
    dollar_volume = (df["close"] * df["volume"]).fillna(0.0).values

    illiq = _calculate_amihud_illiquidity(returns, dollar_volume, window)
    # Scale by 1e12 to make the raw ratio human-readable (e.g., 0.5148 instead of 0.0000)
    return pd.Series(illiq * 1e12, index=df.index, name="amihud_illiq")


def compute_momentum(df: pd.DataFrame,
                     lookback: int = 252,
                     skip: int = 21) -> pd.Series:
    """
    12-Minus-1 Month Cross-Sectional Momentum.

    Skips the most recent *skip* days to avoid short-term reversal contamination.

    Parameters
    ----------
    df       : Daily OHLCV DataFrame with 'close' column.
    lookback : Total lookback in trading days (default 252 ~ 12 months).
    skip     : Recent days excluded from the return window (default 21 ~ 1 month).

    Returns
    -------
    pd.Series named 'momentum_12_1'.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column.")

    shifted_close = df["close"].shift(skip)
    mom = shifted_close / df["close"].shift(lookback) - 1.0
    return pd.Series(mom, index=df.index, name="momentum_12_1")


def compute_low_volatility(df: pd.DataFrame, window: int = 63) -> pd.Series:
    """
    Rolling Low-Volatility Factor (63-day std, sign-flipped).

    A higher score means lower realised volatility (desirable for low-vol premium).

    Parameters
    ----------
    df     : Daily OHLCV DataFrame with 'close' column.
    window : Rolling window in trading days (default 63 ~ 1 quarter).

    Returns
    -------
    pd.Series named 'low_volatility'.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column.")

    returns = df["close"].pct_change()
    vol = -returns.rolling(window=window, min_periods=window // 2).std()
    return pd.Series(vol, index=df.index, name="low_volatility")


# ===========================================================================
# Medium-Term Alpha Factors (swing / position, 30d – 6m horizon)
# ===========================================================================




def compute_quarterly_low_volatility(df: pd.DataFrame,
                                      window: int = 63) -> pd.Series:
    """
    Quarterly Low-Volatility Anomaly Factor.

    The Nifty Low Volatility 50 index historically outperforms the broad market
    due to investor preference for lottery-like payoffs and institutional
    benchmark constraints that force under-allocation to low-risk names.

    Logic
    -----
    vol   = rolling std of daily returns over *window* days
    score = -vol          (sign-flip: higher score == structurally lower vol)

    Parameters
    ----------
    df     : Daily OHLCV DataFrame with 'close' column.
    window : Rolling window in trading days (default 63 ~ 1 quarter).

    Returns
    -------
    pd.Series named 'quarterly_low_vol'.
    NaN returned for initial rows where window is incomplete (min_periods = window//2).
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column.")

    daily_returns = df["close"].pct_change()
    raw_vol = daily_returns.rolling(window=window, min_periods=window // 2).std()
    factor = -raw_vol
    return pd.Series(factor, index=df.index, name="quarterly_low_vol")





def compute_sma100_pullback(df: pd.DataFrame, window: int = 100) -> pd.Series:
    """
    Medium-Term Moving Average Pullback (% Distance from 100-Day SMA).

    Measures how extended or compressed the price is relative to its
    intermediate trend anchor.  In a secular bull market, stocks that are
    moderately above their 100-SMA (but not over-extended) offer superior
    risk-adjusted entry points.

    Logic
    -----
    sma_100 = rolling mean of close over *window* days
    score   = (close - sma_100) / sma_100

    Interpretation
    --------------
    +0.05  → close is 5% above the 100-SMA  (extended; crowded long)
    -0.03  → close is 3% below the 100-SMA  (pullback; potential entry)
     0.00  → close is exactly at the 100-SMA

    Parameters
    ----------
    df     : Daily OHLCV DataFrame with 'close' column.
    window : SMA period in trading days (default 100).

    Returns
    -------
    pd.Series named 'sma100_pullback'.
    NaN returned for initial rows (min_periods = window // 2).
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column.")

    sma = df["close"].rolling(window=window, min_periods=window // 2).mean()
    factor = (df["close"] - sma) / sma
    return pd.Series(factor, index=df.index, name="sma100_pullback")


def compute_vpt_accumulation(df: pd.DataFrame,
                              roll_window: int = 63) -> pd.Series:
    """
    63-Day Institutional Accumulation via Volume-Price Trend (VPT).

    Tracks net smart-money flow by weighting each day's volume by its
    directional price contribution.  The 63-day rolling sum captures a full
    quarter of accumulation / distribution pressure, filtering out intraday
    noise that distorts shorter-term flow measures.

    Logic
    -----
    daily_vpt      = volume * (close - prev_close) / prev_close
    quarterly_vpt  = rolling sum of daily_vpt over *roll_window* days

    Parameters
    ----------
    df          : Daily OHLCV DataFrame with 'close' and 'volume' columns.
    roll_window : Accumulation window in trading days (default 63 ~ 1 quarter).

    Returns
    -------
    pd.Series named 'vpt_accumulation'.
    NaN returned for the first row (prev_close undefined) and initial window rows
    (min_periods = roll_window // 2).

    Notes
    -----
    * Factor values are in native share-volume units and are not normalised.
      For cross-sectional ranking, z-score or rank-normalise before combining
      with other factors.
    * Positive value: net buying pressure over the quarter.
    * Negative value: net selling / distribution pressure over the quarter.
    """
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("DataFrame must contain 'close' and 'volume' columns.")

    prev_close = df["close"].shift(1)
    daily_vpt = df["volume"] * (df["close"] - prev_close) / prev_close
    factor = daily_vpt.rolling(window=roll_window, min_periods=roll_window // 2).sum()
    return pd.Series(factor, index=df.index, name="vpt_accumulation")


# ===========================================================================
# Composite Aggregator
# ===========================================================================

def compute_all_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all price/volume factors and attach them to a copy of *df*.

    Columns added
    -------------
    factor_amihud               -- Amihud liquidity score (sign-flipped: higher = more liquid)
    factor_momentum             -- 12-1 month cross-sectional momentum
    factor_quarterly_lowvol     -- 63-day quarterly low-vol anomaly (sign-flipped)
    factor_sma100_pullback      -- % distance from 100-day SMA
    factor_vpt_accumulation     -- 63-day VPT institutional accumulation

    Parameters
    ----------
    df : Daily OHLCV DataFrame (must contain at minimum 'close' and 'volume').

    Returns
    -------
    pd.DataFrame — a copy of *df* with all factor columns appended.
    """
    out = df.copy()

    # --- Original factors ---
    out["factor_amihud"] = -compute_amihud(df)          # sign-flip: higher = more liquid
    out["factor_momentum"] = compute_momentum(df)

    # --- Medium-term alpha expansion ---
    out["factor_quarterly_lowvol"] = compute_quarterly_low_volatility(df)
    out["factor_sma100_pullback"] = compute_sma100_pullback(df)
    out["factor_vpt_accumulation"] = compute_vpt_accumulation(df)

    return out
