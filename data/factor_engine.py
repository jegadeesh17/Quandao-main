"""
quandao_public/data/factor_engine.py
=======================================
Pure-function factor computation engine for the Quandao multi-factor model.

WHAT THIS MODULE DOES:
    Computes 6 quantitative alpha factors on OHLCV DataFrames.
    Each factor is documented with its academic source and economic intuition.

DESIGN PRINCIPLE:
    Each function: DataFrame in → DataFrame out (with new column added).
    Pure functions — no side effects, no database calls, no global state.
    compute_all_price_factors() runs all 6 in sequence — one-stop call.

FACTOR OVERVIEW:
    1. factor_sharpe_trend      — Volatility-adjusted 90-day momentum
    2. factor_intermediate_mom  — 6-minus-1 month momentum (skip most recent)
    3. factor_amihud            — Amihud illiquidity (inverted → liquidity score)
    4. factor_quarterly_lowvol  — 63-day realized volatility (low-vol anomaly)
    5. factor_sma100_pullback   — Distance from SMA-100 (mean-reversion entry)
    6. factor_vpt_accumulation  — Volume-Price Trend (institutional accumulation)

INTERVIEW ANCHORS:
    "Why skip the last month in momentum?" → Short-term reversal effect (Lo & MacKinlay)
    "Why is Amihud inverted?" → Less liquid stocks score higher on illiquidity → lower score desired
    "Why low-vol as a factor?" → Low-vol anomaly: low-beta stocks outperform CAPM prediction
    "What is VPT?" → Direction-weighted volume accumulation: rising on high vol, falling on low vol

USAGE:
    from quandao_public.data.factor_engine import compute_all_price_factors
    df_with_factors = compute_all_price_factors(df)
"""

import numpy as np
import pandas as pd


# ── Factor 1: Sharpe-Adjusted Trend ────────────────────────────────────────

def factor_sharpe_trend(df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Sharpe-Adjusted Trend (Volatility-Weighted Momentum).

    FORMULA:
        factor = rolling_mean(daily_return, window) / rolling_std(daily_return, window)
        (This is essentially a rolling Sharpe ratio without the risk-free adjustment.)

    ECONOMIC INTUITION:
        A stock that trends up with LOW volatility scores higher than one that
        trends up with HIGH volatility. This captures QUALITY of momentum,
        not just raw return. Preferred over raw momentum for mean-variance portfolios.

    ACADEMIC REFERENCE:
        Inspired by the "Sharpe Momentum" literature. Related to Asness et al. (2013).

    Parameters
    ----------
    df     : pd.DataFrame - OHLCV data with 'close' column
    window : int          - Rolling window in trading days (default 90 ≈ 4 months)

    Returns
    -------
    pd.DataFrame with new column 'factor_sharpe_trend'
    """
    df = df.copy()
    daily_ret = df['close'].pct_change()

    roll_mean = daily_ret.rolling(window=window, min_periods=max(20, window // 3)).mean()
    roll_std  = daily_ret.rolling(window=window, min_periods=max(20, window // 3)).std(ddof=1)

    # Avoid division by zero on flat periods
    df['factor_sharpe_trend'] = np.where(roll_std > 1e-8, roll_mean / roll_std, np.nan)
    return df


# ── Factor 2: Intermediate Momentum ────────────────────────────────────────

def factor_intermediate_momentum(df: pd.DataFrame,
                                   formation_months: int = 6,
                                   skip_months: int = 1) -> pd.DataFrame:
    """
    Intermediate Momentum (6-minus-1 Month Price Return).

    FORMULA:
        factor = (close[t - skip_days] / close[t - formation_days]) - 1

        formation_days = formation_months * 21  (≈ trading days per month)
        skip_days      = skip_months * 21

    WHY SKIP THE LAST MONTH:
        Jegadeesh & Titman (1993) found that 1-month return has REVERSAL properties:
        stocks that went up last month tend to underperform next month.
        By skipping the last month, we isolate the medium-term momentum signal
        and avoid loading on the contaminating short-term reversal.

    ACADEMIC REFERENCE:
        Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers."
        JF 48(1): 65-91. One of the most replicated findings in finance.

    Parameters
    ----------
    df               : pd.DataFrame - OHLCV with 'close' column
    formation_months : int          - Lookback period (default 6 months)
    skip_months      : int          - Recent months to skip (default 1)

    Returns
    -------
    pd.DataFrame with new column 'factor_intermediate_mom'
    """
    df = df.copy()

    formation_days = formation_months * 21
    skip_days      = skip_months * 21

    # Return from formation_days ago to skip_days ago (not to today)
    close_formation = df['close'].shift(formation_days)
    close_skip      = df['close'].shift(skip_days)

    df['factor_intermediate_mom'] = (close_skip / close_formation) - 1.0
    return df


# ── Factor 3: Amihud Illiquidity → Liquidity Factor ────────────────────────

def factor_amihud_liquidity(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Amihud Illiquidity Ratio — transformed to a Liquidity SCORE.

    AMIHUD ILLIQUIDITY FORMULA:
        ILLIQ_t = |daily_return_t| / (Volume_t * Price_t)
        Average over window: ILLIQ = mean(ILLIQ_t, window)

    TRANSFORMATION:
        factor_amihud = -ILLIQ   (negated so HIGHER = MORE LIQUID = BETTER)
        Then cross-sectionally z-scored in the screener.

    ECONOMIC INTUITION:
        Amihud's ratio measures "price impact per rupee of trading volume."
        A stock that moves 1% on ₹10 crore volume is LESS liquid than one
        that moves 1% on ₹1000 crore volume.
        Less liquid stocks carry higher execution risk → we prefer liquid stocks.

    ACADEMIC REFERENCE:
        Amihud (2002), "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects."
        Journal of Financial Markets 5(1): 31-56.

    Parameters
    ----------
    df     : pd.DataFrame - OHLCV data with 'close' and 'volume' columns
    window : int          - Rolling window for illiquidity smoothing (default 21 days ≈ 1 month)

    Returns
    -------
    pd.DataFrame with new column 'factor_amihud' (negative illiquidity = positive liquidity)
    """
    df = df.copy()

    daily_ret = df['close'].pct_change().abs()
    turnover  = df['volume'] * df['close']  # ₹ turnover (volume × price)

    # Illiquidity: |return| / ₹turnover. Add epsilon to avoid division by zero.
    illiq = daily_ret / (turnover + 1e-10)

    # Rolling mean illiquidity
    rolling_illiq = illiq.rolling(window=window, min_periods=max(5, window // 3)).mean()

    # INVERT: higher (less negative) = more liquid = better score
    df['factor_amihud'] = -rolling_illiq
    return df


# ── Factor 4: Quarterly Low Volatility ─────────────────────────────────────

def factor_quarterly_low_volatility(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Quarterly Low Volatility Factor (63-day Realized Volatility, inverted).

    FORMULA:
        realized_vol = std(daily_returns, window=63)
        factor = -realized_vol   (lower vol → higher score)

    THE LOW-VOLATILITY ANOMALY:
        CAPM predicts: higher beta (risk) → higher return.
        REALITY: Low-volatility stocks OUTPERFORM high-volatility stocks on a
        risk-adjusted basis. This is one of the most documented anomalies.

        WHY?: Institutional constraints (benchmark hugging), leverage aversion,
        and lottery-ticket preference cause overcrowding of volatile stocks.

    ACADEMIC REFERENCE:
        Baker, Bradley & Wurgler (2011), "Benchmarks as Limits to Arbitrage."
        Blitz & Van Vliet (2007), "The Volatility Effect."

    Parameters
    ----------
    df     : pd.DataFrame - OHLCV data with 'close' column
    window : int          - Rolling window (63 days ≈ 1 quarter)

    Returns
    -------
    pd.DataFrame with 'factor_quarterly_lowvol' (negative vol = positive factor score)
    """
    df = df.copy()

    daily_ret = df['close'].pct_change()
    realized_vol = daily_ret.rolling(window=window, min_periods=max(10, window // 3)).std(ddof=1)

    # INVERT: lower volatility → higher score
    df['factor_quarterly_lowvol'] = -realized_vol
    return df


# ── Factor 5: SMA-100 Pullback Distance ────────────────────────────────────

def factor_sma100_pullback(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMA-100 Pullback Distance Factor.

    FORMULA:
        sma_100 = rolling_mean(close, 100)
        factor  = (close - sma_100) / sma_100   [dimensionless % distance]

    INTERPRETATION:
        Positive values → stock is above its 100-day average (trending up).
        Values close to 0 → stock has pulled back to its trend line.
        Large positive values → possibly overbought (use with RSI filter).

    ECONOMIC INTUITION:
        Stocks that are close to (but above) SMA-100 present better entry points
        than those far above. This factor acts as a MEAN-REVERSION TIMING signal
        within an uptrend — buy the dip without fighting the trend.

    USAGE IN SCREENER:
        The screener's Stage 3 filter flags stocks with pullback between 0% and 5%
        as "Optimal Support Entry" (the ideal buy zone).

    Returns
    -------
    pd.DataFrame with 'factor_sma100_pullback'
    """
    df = df.copy()
    sma_100 = df['close'].rolling(window=100, min_periods=50).mean()
    df['factor_sma100_pullback'] = (df['close'] - sma_100) / (sma_100 + 1e-8)
    return df


# ── Factor 6: VPT Volume-Price Trend Accumulation ──────────────────────────

def factor_vpt_accumulation(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    VPT (Volume-Price Trend) Institutional Accumulation Factor.

    FORMULA:
        VPT_t = VPT_{t-1} + Volume_t × (Close_t - Close_{t-1}) / Close_{t-1}

        Simplified: VPT_t = cumsum(Volume × daily_return)

    VPT vs OBV:
        OBV (On-Balance Volume): add FULL volume on up-days, subtract on down-days.
        VPT: weight volume by the MAGNITUDE of the price change.
        VPT is more nuanced — a 5% up day gets 5x the VPT of a 1% up day.

    WHAT VPT DETECTS:
        Rising VPT = institutional accumulation: big money is buying on up-days.
        Falling VPT = distribution: institutions selling into price strength.
        VPT divergence (price rises but VPT falls) is a classic bearish warning.

    ACADEMIC REFERENCE:
        Related to Granville's OBV (1963). VPT variant by Buff Dormeier.

    Parameters
    ----------
    df     : pd.DataFrame - OHLCV data with 'close' and 'volume' columns
    window : int          - Rolling window to z-score VPT (default 63 days)

    Returns
    -------
    pd.DataFrame with 'factor_vpt_accumulation'
    """
    df = df.copy()

    daily_ret = df['close'].pct_change().fillna(0)
    vpt_daily = df['volume'] * daily_ret

    # Cumulative VPT from start
    vpt_cumsum = vpt_daily.cumsum()

    # Use rolling 63-day change in VPT to measure RECENT accumulation pace
    df['factor_vpt_accumulation'] = vpt_cumsum.diff(window)
    return df


# ── Factor 7: EMA-21 Proximity ──────────────────────────────────────────────

def factor_ema_proximity(df: pd.DataFrame, span: int = 21) -> pd.DataFrame:
    """
    EMA-21 Proximity Factor — distance of close from the 21-day EMA.

    FORMULA:
        ema_21 = EWM(close, span=21)
        factor = (close - ema_21) / ema_21   [dimensionless % distance]

    SWING TRADING INTERPRETATION:
        Positive values → stock is above EMA-21 (uptrend).
        Values near 0   → ideal pullback entry zone on the EMA.
        Large positive  → extended from EMA, higher reversion risk.
        Negative values → price has dipped below EMA-21 (caution).

    ECONOMIC INTUITION:
        EMA-21 is the short-term institutional "cost basis" reference.
        Entries within 0–3% above EMA-21 provide tight stops and
        high-probability mean-reversion trades in trending stocks.

    Parameters
    ----------
    df   : pd.DataFrame - OHLCV data with 'close' column
    span : int          - EMA span (default 21 days)

    Returns
    -------
    pd.DataFrame with new column 'factor_ema_proximity'
    """
    df = df.copy()
    ema = df['close'].ewm(span=span, adjust=False).mean()
    df['factor_ema_proximity'] = (df['close'] - ema) / (ema + 1e-8)
    return df


# ── Factor 8: Volume Surge ───────────────────────────────────────────────────

def factor_volume_surge(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Volume Surge Factor — ratio of current volume to 20-day Volume Moving Average.

    FORMULA:
        VMA_20 = rolling_mean(volume, window=20)
        factor = volume / VMA_20

    INTERPRETATION:
        factor = 1.0  → average volume day (baseline)
        factor > 1.5  → elevated volume: institutional participation likely
        factor > 2.0  → high conviction breakout or reversal bar
        factor < 0.7  → low-volume drift: unreliable price action

    WHY VOLUME CONFIRMS SIGNALS:
        Price moves on above-average volume carry more weight because they
        represent genuine supply/demand imbalance — not just algorithmic
        noise or thin-market drift. A bullish engulfing on 2× volume is
        qualitatively different from one on 0.5× volume.

    USAGE IN SCREENER:
        Used as both a factor in the composite score AND a binary gate
        (vol_confirmed = factor_volume_surge > 1.0 on the signal bar).

    Parameters
    ----------
    df     : pd.DataFrame - OHLCV data with 'volume' column
    window : int          - Rolling window for VMA (default 20 days)

    Returns
    -------
    pd.DataFrame with new column 'factor_volume_surge'
    """
    df = df.copy()
    vma = df['volume'].rolling(window=window, min_periods=max(5, window // 3)).mean()
    df['factor_volume_surge'] = df['volume'] / (vma + 1e-8)
    return df


# ── Master Factor Computation ────────────────────────────────────────────────

def compute_all_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all 8 alpha factors in sequence on a single OHLCV DataFrame.

    This is the one-stop function called by the screener and dashboard.
    Each factor adds one column to the DataFrame.

    INPUT columns required:  time, open, high, low, close, volume
    OUTPUT columns added:    factor_sharpe_trend, factor_intermediate_mom,
                             factor_amihud, factor_quarterly_lowvol,
                             factor_sma100_pullback, factor_vpt_accumulation,
                             factor_ema_proximity, factor_volume_surge

    Parameters
    ----------
    df : pd.DataFrame - Clean OHLCV data (sorted by time, no NaN in OHLCV columns)

    Returns
    -------
    pd.DataFrame with original columns + 8 new factor columns.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Run all factors in order (each is pure, no conflicts)
    df = factor_sharpe_trend(df)
    df = factor_intermediate_momentum(df)
    df = factor_amihud_liquidity(df)
    df = factor_quarterly_low_volatility(df)
    df = factor_sma100_pullback(df)
    df = factor_vpt_accumulation(df)
    df = factor_ema_proximity(df)
    df = factor_volume_surge(df)

    return df

