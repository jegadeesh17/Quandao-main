"""
quandao_public/strategies/multi_factor.py
============================================
Multi-factor composite scoring, IC-weighted combination, sector neutralization,
and PCA statistical arbitrage signal generation.

THIS MODULE IS THE QUANTITATIVE BRAIN OF THE SCREENER.

WHAT EACH FUNCTION DOES:

    neutralize_factor()
        Remove sector-level mean from factor scores.
        Prevents Banking or IT rallies from dominating the cross-sectional ranking.
        "Sector-neutral" means scores are relative within their own industry group.

    compute_ic()
        Information Coefficient: rank correlation between factor and forward return.
        The PRIMARY measure of factor quality. Good IC = 0.05–0.10.

    compute_rolling_ic()
        IC computed over a rolling time window.
        Used to detect whether a factor's predictive power is stable or decaying.

    compute_ic_decay()
        IC at 1, 5, 10, 21, 42, 63-day forward return horizons.
        Reveals signal half-life → tells you optimal holding period.

    ic_weighted_combination()
        Weight each factor by its recent IC stability (ICIR > 0.5 threshold).
        Better than equal-weighting: rewards currently-predictive factors.

    generate_multi_factor_composite()
        Weighted sum of normalized factor scores.

    generate_directional_signal_from_composite()
        Long top quintile, short/avoid bottom quintile.

    generate_pca_stat_arb_signals()
        PCA-based statistical arbitrage: long under-valued, short over-valued
        stocks relative to their principal component factor model.

USAGE:
    from quandao_public.strategies.multi_factor import (
        generate_multi_factor_composite,
        generate_directional_signal_from_composite,
        neutralize_factor,
        generate_pca_stat_arb_signals,
    )
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── Sector Neutralization ────────────────────────────────────────────────────

def neutralize_factor(df: pd.DataFrame,
                       factor_col: str,
                       group_col: str = 'Sector') -> pd.Series:
    """
    Sector-neutralize a factor column by removing the within-sector mean.

    WHY WE NEUTRALIZE:
        Without neutralization, if the Banking sector is rallying,
        ALL banking stocks score high on momentum — not because of stock-specific
        alpha, but because of sector beta. The screener would just return a basket
        of banking stocks, exposing us to sector concentration risk.

        Neutralization computes scores RELATIVE to the sector mean:
            neutralized = raw_score - sector_average(raw_score)
        A score of +0.5 now means "this stock is 0.5 above its sector average."

    FORMULA:
        group_means = df.groupby(group_col)[factor_col].transform('mean')
        neutralized = df[factor_col] - group_means

    Parameters
    ----------
    df         : pd.DataFrame - DataFrame with factor_col and group_col
    factor_col : str          - Column name of the factor to neutralize
    group_col  : str          - Column to group by (default 'Sector')

    Returns
    -------
    pd.Series : Neutralized factor values (same index as df)
    """
    if factor_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    if group_col not in df.columns:
        # No grouping available — return z-scored factor as fallback
        mu  = df[factor_col].mean()
        std = df[factor_col].std()
        return (df[factor_col] - mu) / (std + 1e-8)

    group_means = df.groupby(group_col)[factor_col].transform('mean')
    return df[factor_col] - group_means


# ── Information Coefficient ──────────────────────────────────────────────────

def compute_ic(factor_series: pd.Series,
               forward_return_series: pd.Series,
               method: str = 'spearman') -> float:
    """
    Information Coefficient (IC): predictive correlation between factor and return.

    FORMULA:
        IC = Spearman rank correlation(factor_t, forward_return_{t+1})

    WHY SPEARMAN (NOT PEARSON):
        Factor values and returns both have fat tails and outliers.
        Spearman is robust to outliers — it operates on RANKS, not raw values.
        Pearson IC would be dominated by extreme observations.

    WHAT IC MEANS:
        IC = 0.05 → Factor has modest but positive predictive power.
        IC = 0.10 → Strong factor (top-decile institutional quality).
        IC > 0.15 → Exceptional (verify for data mining before celebrating).
        IC < 0    → Factor predicts the WRONG direction → invert the sign.
        IC ≈ 0    → Factor has no predictive power → discard.

    Parameters
    ----------
    factor_series         : pd.Series - Factor values at time t
    forward_return_series : pd.Series - Forward returns at time t+1 (or +N)
    method                : str       - 'spearman' (recommended) or 'pearson'

    Returns
    -------
    float : IC value in [-1, 1]. NaN if insufficient data.
    """
    # Align and drop NaN
    combined = pd.DataFrame({'factor': factor_series, 'fwd_ret': forward_return_series})
    combined = combined.dropna()

    if len(combined) < 5:
        return np.nan

    if method == 'spearman':
        ic, _ = spearmanr(combined['factor'], combined['fwd_ret'])
    else:
        ic = combined['factor'].corr(combined['fwd_ret'])

    return float(ic)


# ── Rolling IC Over Time ─────────────────────────────────────────────────────

def compute_rolling_ic(panel_df: pd.DataFrame,
                        factor_col: str,
                        return_col: str = 'fwd_return',
                        window: int = 63,
                        group_col: str = 'time') -> pd.Series:
    """
    Rolling Information Coefficient over time periods.

    Computes cross-sectional IC (across stocks) at each time period,
    then takes a rolling mean over `window` periods.

    Parameters
    ----------
    panel_df   : pd.DataFrame - Panel data: rows = (symbol, time), cols = factor + returns
    factor_col : str          - Factor column name
    return_col : str          - Forward return column name
    window     : int          - Rolling window for IC smoothing (default 63 = 1 quarter)
    group_col  : str          - Column defining time periods (default 'time')

    Returns
    -------
    pd.Series : Rolling IC indexed by time
    """
    def cross_section_ic(group):
        valid = group.dropna(subset=[factor_col, return_col])
        if len(valid) < 5:
            return np.nan
        ic, _ = spearmanr(valid[factor_col], valid[return_col])
        return float(ic)

    period_ics = panel_df.groupby(group_col).apply(cross_section_ic)
    rolling_ic = period_ics.rolling(window=window, min_periods=max(5, window // 4)).mean()
    return rolling_ic


# ── IC Decay Analysis ────────────────────────────────────────────────────────

def compute_ic_decay(panel_df: pd.DataFrame,
                      factor_col: str,
                      close_col: str = 'close',
                      horizons: list = None) -> pd.DataFrame:
    """
    IC Decay Analysis — how fast does a factor's predictive power decay over time?

    WHAT IC DECAY TELLS YOU:
        IC at 1 day → best for very short-term signals
        IC drops to 50% at 10 days → signal half-life = 10 days
        IC near zero by 30 days → factor is useless beyond 1 month holding period

        This directly tells you:
            - Optimal holding period (before IC decays to noise)
            - Rebalancing frequency (rebalance every half-life period)

    FORMULA:
        For each horizon h (1, 5, 10, 21, 42, 63 days):
            fwd_return_h = close.shift(-h) / close - 1
            IC_h = cross-sectional Spearman(factor, fwd_return_h)

    Parameters
    ----------
    panel_df   : pd.DataFrame - Panel with factor_col, close_col, indexed or grouped by time+symbol
    factor_col : str          - Factor column name
    close_col  : str          - Close price column
    horizons   : list of int  - Forward return horizons to test (default [1,5,10,21,42,63])

    Returns
    -------
    pd.DataFrame : IC values at each horizon.
        Columns: ['horizon_days', 'ic_mean', 'ic_std', 'icir', 'n_observations']
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 42, 63]

    results = []
    close   = panel_df[close_col]
    factor  = panel_df[factor_col]

    for h in horizons:
        fwd_return = (close.shift(-h) / close - 1).dropna()

        aligned = pd.DataFrame({
            'factor':     factor,
            'fwd_return': fwd_return,
        }).dropna()

        if len(aligned) < 10:
            results.append({
                'horizon_days':  h,
                'ic_mean':       np.nan,
                'ic_std':        np.nan,
                'icir':          np.nan,
                'n_observations': 0,
            })
            continue

        ic, _ = spearmanr(aligned['factor'], aligned['fwd_return'])
        results.append({
            'horizon_days':   h,
            'ic_mean':        round(float(ic), 6),
            'ic_std':         np.nan,  # Requires panel groupby for time-series std
            'icir':           np.nan,
            'n_observations': len(aligned),
        })

    return pd.DataFrame(results)


# ── IC-Weighted Composite ────────────────────────────────────────────────────

def ic_weighted_combination(factor_df: pd.DataFrame,
                              rolling_ic_dict: dict,
                              icir_threshold: float = 0.3) -> pd.Series:
    """
    Combine factors using IC-weighted (ICIR-filtered) weights.

    ALGORITHM:
        1. For each factor, compute its recent ICIR (IC / std of IC).
        2. Factors with ICIR < threshold get ZERO weight (they are noise).
        3. Surviving factors are weighted proportionally to their ICIR.
        4. Composite = sum(factor * normalized_weight)

    WHY IC-WEIGHTING BEATS EQUAL WEIGHTING:
        Equal weighting treats a factor with IC=0.08 the same as one with IC=0.001.
        IC-weighting rewards factors that are currently predictive and
        reduces weight on factors that have become noisy.

    Parameters
    ----------
    factor_df       : pd.DataFrame  - DataFrame with one column per factor
    rolling_ic_dict : dict          - {factor_name: rolling_IC_series}
    icir_threshold  : float         - Minimum ICIR to include a factor (default 0.3)

    Returns
    -------
    pd.Series : Composite score (IC-weighted sum of factor scores)
    """
    weights = {}
    for factor_name, ic_series in rolling_ic_dict.items():
        if factor_name not in factor_df.columns:
            continue
        ic_vals = ic_series.dropna()
        if len(ic_vals) < 3:
            continue
        ic_mean = float(ic_vals.mean())
        ic_std  = float(ic_vals.std())
        icir    = ic_mean / ic_std if ic_std > 1e-8 else 0.0
        weights[factor_name] = max(0.0, icir) if icir >= icir_threshold else 0.0

    total_weight = sum(weights.values())
    if total_weight < 1e-8:
        # All factors failed ICIR threshold → equal weight as fallback
        available = [c for c in factor_df.columns if c in rolling_ic_dict]
        total_weight = len(available)
        weights = {c: 1.0 for c in available}

    composite = pd.Series(0.0, index=factor_df.index)
    for factor_name, w in weights.items():
        if factor_name in factor_df.columns and total_weight > 0:
            # Cross-sectional z-score before weighting
            col = factor_df[factor_name]
            z_scored = (col - col.mean()) / (col.std() + 1e-8)
            composite += z_scored * (w / total_weight)

    return composite


# ── Composite Score Generation ───────────────────────────────────────────────

def generate_multi_factor_composite(df: pd.DataFrame,
                                      weights: dict) -> pd.Series:
    """
    Weighted composite score from a dictionary of factor weights.

    Parameters
    ----------
    df      : pd.DataFrame - DataFrame with factor columns
    weights : dict         - {factor_col_name: weight} (weights need not sum to 1)

    Returns
    -------
    pd.Series : Composite factor score
    """
    total_weight = sum(abs(w) for w in weights.values())
    if total_weight < 1e-8:
        return pd.Series(0.0, index=df.index)

    composite = pd.Series(0.0, index=df.index)
    for factor_col, weight in weights.items():
        if factor_col not in df.columns:
            continue
        col = df[factor_col].fillna(0.0)
        # Z-score each factor before combining (prevents scale differences from dominating)
        z = (col - col.mean()) / (col.std() + 1e-8)
        composite += z * (weight / total_weight)

    return composite


# ── Directional Signal ───────────────────────────────────────────────────────

def generate_directional_signal_from_composite(df: pd.DataFrame,
                                                 composite_col: str = 'composite_score',
                                                 top_pct: float = 0.20,
                                                 bottom_pct: float = 0.20) -> pd.Series:
    """
    Convert composite scores to directional signals: LONG / SHORT / NEUTRAL.

    Top quintile (top 20%) → +1 (LONG)
    Bottom quintile (bottom 20%) → -1 (SHORT)
    Middle 60% → 0 (NEUTRAL)

    Parameters
    ----------
    df            : pd.DataFrame - DataFrame with composite_col
    composite_col : str          - Column with composite scores
    top_pct       : float        - Top fraction to go long
    bottom_pct    : float        - Bottom fraction to avoid/short

    Returns
    -------
    pd.Series : Signal values (-1, 0, +1)
    """
    scores   = df[composite_col]
    q_high   = scores.quantile(1.0 - top_pct)
    q_low    = scores.quantile(bottom_pct)

    signal = pd.Series(0, index=df.index)
    signal[scores >= q_high] =  1   # LONG
    signal[scores <= q_low]  = -1   # SHORT / Avoid

    return signal


# ── PCA Statistical Arbitrage ────────────────────────────────────────────────

def generate_pca_stat_arb_signals(panel_df: pd.DataFrame,
                                   n_components: int = 3,
                                   zscore_threshold: float = 2.0,
                                   return_col: str = 'close') -> pd.DataFrame:
    """
    PCA-based Statistical Arbitrage Signal.

    THE IDEA:
        Stock prices are driven by common factors (market, sector, style).
        PCA extracts these common factors from the returns matrix.
        The RESIDUAL = actual return - PCA-explained return represents
        idiosyncratic (stock-specific) price dislocation.

        A highly NEGATIVE residual → stock fell more than its factor model predicts
        → potentially oversold → LONG opportunity.

        A highly POSITIVE residual → stock rose more than its factor model predicts
        → potentially overbought → SHORT/avoid.

    ALGORITHM:
        1. Build returns matrix: rows = dates, columns = symbols
        2. Standardize the returns matrix
        3. Fit PCA with n_components → get n factor loadings
        4. Reconstruct returns from PCA components
        5. Residuals = actual - reconstructed
        6. Z-score residuals cross-sectionally
        7. Signal: z < -threshold → LONG, z > threshold → SHORT

    Parameters
    ----------
    panel_df         : pd.DataFrame - Panel: must have 'symbol', 'time', return_col
    n_components     : int          - Number of PCA components (3–5 is standard)
    zscore_threshold : float        - Z-score cutoff for signal (default 2.0 = 2 sigma)
    return_col       : str          - Price/return column to use

    Returns
    -------
    pd.DataFrame : Per-symbol signals with columns [symbol, residual_zscore, signal]
    """
    try:
        # Pivot: rows = time, columns = symbol, values = return_col
        if 'symbol' not in panel_df.columns or 'time' not in panel_df.columns:
            return pd.DataFrame()

        pivot = panel_df.pivot_table(
            index='time', columns='symbol', values=return_col
        )

        # Convert to returns if we got prices
        if return_col in ('close', 'open', 'price'):
            pivot = pivot.pct_change().dropna(how='all')

        # Drop symbols with too many NaN
        pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.8))
        pivot = pivot.fillna(0.0)

        if pivot.shape[1] < n_components + 2:
            return pd.DataFrame()

        # PCA on the returns matrix
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pivot.values)

        pca = PCA(n_components=min(n_components, pivot.shape[1] - 1))
        components = pca.fit_transform(X_scaled)       # (T, n_components)
        loadings    = pca.components_                   # (n_components, n_symbols)

        # Reconstruct: what PCA "explains" for each stock
        reconstructed = components @ loadings           # (T, n_symbols)
        residuals_raw = X_scaled - reconstructed        # (T, n_symbols) — idiosyncratic

        # Use the MOST RECENT row of residuals as the signal
        latest_residuals = residuals_raw[-1]            # (n_symbols,)

        # Z-score across the cross-section (latest period)
        mu    = np.mean(latest_residuals)
        sigma = np.std(latest_residuals)
        z_scores = (latest_residuals - mu) / (sigma + 1e-8)

        symbols = pivot.columns.tolist()
        signals = []
        for sym, z in zip(symbols, z_scores):
            if z <= -zscore_threshold:
                sig = 'LONG'    # Oversold vs factor model
            elif z >= zscore_threshold:
                sig = 'SHORT'   # Overbought vs factor model
            else:
                sig = 'NEUTRAL'

            signals.append({
                'symbol':          sym,
                'residual_zscore': round(float(z), 4),
                'signal':          sig,
                'pca_explained_var': round(float(sum(pca.explained_variance_ratio_) * 100), 2),
            })

        return pd.DataFrame(signals).sort_values('residual_zscore')

    except Exception as e:
        print(f"[generate_pca_stat_arb_signals] Error: {e}")
        return pd.DataFrame()

