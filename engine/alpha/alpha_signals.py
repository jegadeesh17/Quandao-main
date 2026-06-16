"""
quandao_public/strategies/alpha_signals.py
=============================================
Statistical signal generation: GARCH volatility, PCA residuals, and IC decay.

WHAT THIS MODULE PROVIDES:
    1. estimate_garch_volatility()  — EGARCH(1,1) conditional volatility
    2. compute_pca_residuals()      — PCA factor model residuals for stat-arb
    3. compute_ic_decay_panel()     — IC at multiple forward horizons on panel data

WHY GARCH FOR REGIME DETECTION:
    GARCH (Generalized Autoregressive Conditional Heteroscedasticity) models
    TIME-VARYING VOLATILITY — the fact that volatile periods cluster together.
    (A 2% crash day is more likely to be followed by another volatile day than
    a calm 0.1% day is.)

    We use EGARCH (Exponential GARCH) which:
        a) Avoids the non-negativity constraint of regular GARCH
        b) Captures the ASYMMETRIC volatility effect: negative shocks (crashes)
           cause MORE volatility than equivalent positive shocks.

    RISK MANAGEMENT USE:
        When cond_vol (conditional volatility) > threshold (e.g. 2.5%/day):
            → REDUCE position sizes by 50%
            → SHIFT options strategy from buying to selling spreads
        This prevents drawdown cascades during volatile regimes.

USAGE:
    from quandao_public.engine.alpha.alpha_signals import (
        estimate_garch_volatility, compute_pca_residuals
    )

    garch_df = estimate_garch_volatility(daily_df)
    print(garch_df['cond_vol'].iloc[-1])  # Today's conditional vol in % daily
"""

import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: GARCH Volatility Estimation
# ══════════════════════════════════════════════════════════════════════════════

def estimate_garch_volatility(df: pd.DataFrame,
                               p: int = 1,
                               q: int = 1,
                               dist: str = 'normal') -> pd.DataFrame:
    """
    Estimate conditional volatility using EGARCH(p, q) on daily returns.

    THE EGARCH MODEL:
        ln(σ²_t) = ω + α * z_{t-1} + γ * |z_{t-1}| + β * ln(σ²_{t-1})

        Where:
            σ_t   = conditional standard deviation at time t
            z_t   = ε_t / σ_t  (standardized shock)
            ω, α, β, γ = model parameters fitted to data
            α captures the asymmetry: negative z (crash) → more vol than positive z

    PRACTICAL OUTPUT:
        'cond_vol' column = conditional volatility in % per day.
        e.g. cond_vol = 1.5 means the model predicts 1.5% daily vol tomorrow.

    REGIME THRESHOLDS (from Quandao config):
        Normal regime: cond_vol < 1.5%/day
        Elevated:      1.5% < cond_vol < 2.5%/day
        High regime:   cond_vol > 2.5%/day → scale down positions

    REQUIRES: `arch` library (pip install arch)
    Falls back to rolling std if arch is not installed.

    Parameters
    ----------
    df   : pd.DataFrame - Daily OHLCV data with 'close' column
    p    : int          - ARCH order (lag of squared shocks)
    q    : int          - GARCH order (lag of conditional variance)
    dist : str          - Error distribution: 'normal', 'studentst', 'skewt'

    Returns
    -------
    pd.DataFrame : Original df + columns ['log_return', 'cond_vol']
        cond_vol is in % daily units (e.g. 1.5 = 1.5%/day)
    """
    df = df.copy()

    # Compute log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    returns = df['log_return'].dropna()

    try:
        from arch import arch_model

        # Fit EGARCH(p, q) — uses MLE to estimate parameters
        model = arch_model(
            returns * 100,   # Scale to percentage returns for numerical stability
            vol='EGARCH',
            p=p,
            q=q,
            dist=dist,
            mean='Constant',  # Constant mean (appropriate for returns)
        )

        # Fit with quiet mode to suppress output
        fit = model.fit(disp='off', show_warning=False)

        # Extract conditional volatility (in percentage terms)
        cond_vol = fit.conditional_volatility  # % per day

        # Map back to df index
        cond_vol_series = pd.Series(np.nan, index=df.index)
        cond_vol_series.loc[returns.index] = cond_vol.values
        df['cond_vol'] = cond_vol_series

    except ImportError:
        # Fallback: rolling standard deviation of log returns (not a GARCH model,
        # but gives a reasonable volatility estimate for the screener)
        rolling_vol = returns.rolling(window=21, min_periods=5).std() * 100
        cond_vol_series = pd.Series(np.nan, index=df.index)
        cond_vol_series.loc[returns.index] = rolling_vol.values
        df['cond_vol'] = cond_vol_series
        logger.warning("`arch` library not installed; falling back to 21-day rolling std")


    except Exception as e:
        # GARCH fit can fail on very short or flat series
        rolling_vol = returns.rolling(window=21, min_periods=5).std() * 100
        cond_vol_series = pd.Series(np.nan, index=df.index)
        cond_vol_series.loc[returns.index] = rolling_vol.values
        df['cond_vol'] = cond_vol_series
        logger.error(
            "EGARCH fit failed for series of length %d: %s",
            len(returns), e, exc_info=True,
        )


    return df


def classify_vol_regime(cond_vol: float,
                          low_threshold: float = 1.5,
                          high_threshold: float = 2.5) -> str:
    """
    Classify the current volatility regime based on GARCH conditional vol.

    Parameters
    ----------
    cond_vol       : float - Conditional volatility in % per day (from GARCH)
    low_threshold  : float - Below this = 'LOW' regime (default 1.5%/day)
    high_threshold : float - Above this = 'HIGH' regime (default 2.5%/day)

    Returns
    -------
    str : 'LOW', 'NORMAL', or 'HIGH'

    TRADING IMPLICATIONS:
        LOW    → Full position sizes, option buying allowed
        NORMAL → Standard risk management
        HIGH   → Reduce position sizes by 50%, prefer option selling (credit spreads)
    """
    if np.isnan(cond_vol):
        return 'UNKNOWN'
    if cond_vol < low_threshold:
        return 'LOW'
    elif cond_vol < high_threshold:
        return 'NORMAL'
    else:
        return 'HIGH'


def vol_scaled_position_multiplier(cond_vol: float,
                                    target_vol: float = 1.5) -> float:
    """
    Position size multiplier based on volatility targeting.

    FORMULA:
        multiplier = target_vol / current_vol

    EXAMPLES:
        Current vol = 1.5%/day, target = 1.5% → multiplier = 1.0 (full size)
        Current vol = 3.0%/day, target = 1.5% → multiplier = 0.5 (half size)
        Current vol = 0.5%/day, target = 1.5% → multiplier = 3.0 (cap at 2.0)

    Parameters
    ----------
    cond_vol   : float - Current conditional vol in % per day
    target_vol : float - Target daily vol (default 1.5%/day)

    Returns
    -------
    float : Position multiplier capped at [0.25, 2.0]
    """
    if cond_vol <= 0 or np.isnan(cond_vol):
        return 1.0
    raw_mult = target_vol / cond_vol
    return round(float(np.clip(raw_mult, 0.25, 2.0)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PCA Residuals for Statistical Arbitrage
# ══════════════════════════════════════════════════════════════════════════════

def compute_pca_residuals(panel_df: pd.DataFrame,
                           n_components: int = 3,
                           price_col: str = 'close',
                           symbol_col: str = 'symbol',
                           time_col: str = 'time') -> pd.DataFrame:
    """
    Compute PCA-based idiosyncratic residuals for statistical arbitrage.

    THEORY:
        Stock returns can be decomposed into:
            return = common_factors_component + idiosyncratic_residual

        Common factors (extracted by PCA) capture: market beta, sector beta,
        style factors (value, momentum, quality).

        The RESIDUAL is what PCA cannot explain — pure idiosyncratic movement.
        A large negative residual = stock fell for stock-specific reasons
        while its factor model says it should have gone sideways → LONG signal.

    STEPS:
        1. Pivot: returns matrix (T×N) where T=time, N=symbols
        2. Standardize across the cross-section
        3. PCA: extract n_components principal components
        4. Residual = standardized_returns - PCA_reconstruction
        5. Return the most recent residuals as z-scores

    Parameters
    ----------
    panel_df     : pd.DataFrame - Long-format panel with symbol, time, price_col
    n_components : int          - PCA components (3–5 is standard for equity)
    price_col    : str          - Column with prices or returns
    symbol_col   : str          - Column with symbol identifiers
    time_col     : str          - Column with timestamps

    Returns
    -------
    pd.DataFrame : Columns [symbol, residual, residual_zscore, var_explained]
    """
    try:
        # Pivot to wide format: rows = time, cols = symbols
        pivot = panel_df.pivot_table(
            index=time_col,
            columns=symbol_col,
            values=price_col,
        )

        # Convert to returns if we got prices
        if price_col in ('close', 'open', 'price', 'adj_close'):
            pivot = pivot.pct_change().dropna(how='all')

        # Drop symbols with > 20% missing
        pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.8))
        pivot = pivot.fillna(0.0)

        n_syms = pivot.shape[1]
        if n_syms < n_components + 2:
            return pd.DataFrame()

        # Standardize
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(pivot.values)   # shape: (T, N)

        # PCA fit
        pca = PCA(n_components=min(n_components, n_syms - 1))
        scores     = pca.fit_transform(X_scaled)         # (T, n_components)
        loadings   = pca.components_                     # (n_components, N)

        # Reconstruct: the part of returns explained by PCA factors
        X_reconstructed = scores @ loadings              # (T, N)
        residuals_all   = X_scaled - X_reconstructed     # (T, N)

        # Use the LATEST time period for the signal
        latest_residuals = residuals_all[-1]             # (N,)
        mu    = np.mean(latest_residuals)
        sigma = np.std(latest_residuals)
        z_scores = (latest_residuals - mu) / (sigma + 1e-8)

        var_explained = float(sum(pca.explained_variance_ratio_) * 100)
        symbols = pivot.columns.tolist()

        result = pd.DataFrame({
            symbol_col:          symbols,
            'residual':          [round(float(r), 6) for r in latest_residuals],
            'residual_zscore':   [round(float(z), 4) for z in z_scores],
            'var_explained_pct': [round(var_explained, 2)] * len(symbols),
        })

        return result.sort_values('residual_zscore').reset_index(drop=True)

    except Exception as e:
        logger.error("compute_pca_residuals failed: %s", e, exc_info=True)
        return pd.DataFrame()


