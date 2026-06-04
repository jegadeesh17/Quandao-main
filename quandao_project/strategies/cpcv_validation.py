"""
quandao_project/strategies/cpcv_validation.py
===============================================
Combinatorial Purged Cross-Validation (CPCV) and Deflated Sharpe Ratio (DSR).

⭐ THIS IS THE STAR MODULE — the one that separates you from 95% of candidates. ⭐

WHAT THIS MODULE IMPLEMENTS:
    1. CPCV       — Generate combinatorial train/test splits with temporal purging
    2. run_cpcv_backtest — Compute IS and OOS Sharpe across all CPCV paths
    3. compute_dsr — Deflated Sharpe Ratio (corrects for multiple testing bias)
    4. compute_prob_dsr — P(true edge) from DSR

THEORY (Interview-Ready Explanations):

    WHY CPCV BEATS K-FOLD CV:
        Standard K-Fold violates the IID assumption — financial returns are
        autocorrelated. A test fold adjacent to a train fold LEAKS information
        through autocorrelated returns (look-ahead bias by proxy).
        CPCV fixes this with:
            (a) PURGING: removes training samples within an "embargo" zone
                adjacent to each test fold boundary.
            (b) COMBINATORIAL PATHS: tests ALL C(k,p) train/test combinations,
                not just sequential folds — gives a DISTRIBUTION of OOS Sharpes,
                not just one point estimate.
            (c) PBO: Probability of Backtest Overfitting = fraction of paths
                where the best IS configuration underperforms OOS.

    WHY DSR OVER SHARPE:
        If you test 20 strategy formulations, the expected maximum Sharpe from
        pure luck (null hypothesis) is NOT zero — it's sqrt(2 * log(20)) ≈ 2.4.
        DSR penalizes for this: it asks "is your Sharpe statistically greater
        than what you'd expect from 20 random coin flips?"

ACADEMIC REFERENCES:
    - López de Prado (2018), "Advances in Financial Machine Learning", Chapter 12
    - Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", JPM
    - Bailey, Borwein, López de Prado, Zhu (2015), "The Probability of Backtest
      Overfitting", JCAM

USAGE:
    from quandao_project.strategies.cpcv_validation import (
        run_cpcv_backtest, compute_dsr, compute_prob_dsr
    )

    results = run_cpcv_backtest(portfolio_returns, n_splits=6, n_test_splits=2)
    dsr = compute_dsr(results['mean_oos_sharpe'], num_trials=15, returns_series=portfolio_returns)
    prob_edge = compute_prob_dsr(dsr)
"""

from itertools import combinations
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CPCV Split Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_cpcv_splits(n_samples: int,
                          n_splits: int,
                          n_test_splits: int,
                          embargo_pct: float = 0.01) -> list:
    """
    Generate all C(n_splits, n_test_splits) combinatorial purged train/test splits.

    ALGORITHM IN PLAIN ENGLISH:
        1. Divide n_samples indices into n_splits equal sequential blocks.
           e.g. n_samples=1000, n_splits=5 → blocks of 200 each.

        2. Generate all C(5,2) = 10 combinations of 2 test blocks.
           Each combination defines one CPCV "path."

        3. For each path:
           - Test  = indices in the 2 chosen blocks
           - Train = all other indices (3 blocks worth)
           - Embargo = indices within embargo_size of any test block boundary
             are REMOVED from train to prevent leakage through autocorrelation.

    THE EMBARGO EXPLAINED:
        If your returns overlap (e.g. you compute 5-day forward returns),
        a return at the boundary of train/test will contaminate the train label.
        Embargo removes ~1% of samples around each boundary to break this leakage.

    Parameters
    ----------
    n_samples      : int   - Total number of observations
    n_splits       : int   - k: number of sequential blocks (recommend 4–8)
    n_test_splits  : int   - p: number of blocks held as test (recommend 1–3)
    embargo_pct    : float - Fraction of block to purge around boundaries (default 0.01 = 1%)

    Returns
    -------
    list of (train_indices, test_indices) tuples — one per CPCV path.
    Total paths = C(n_splits, n_test_splits)
    """
    # Step 1: Sequential block construction
    indices = np.arange(n_samples)
    blocks  = np.array_split(indices, n_splits)

    # Embargo: number of samples to purge around each test boundary
    embargo_size = max(1, int(n_samples * embargo_pct / n_splits))

    splits = []

    # Step 2: All combinations of test blocks
    for test_block_combo in combinations(range(n_splits), n_test_splits):

        # Test indices: all samples in the selected blocks
        test_indices = np.concatenate([blocks[i] for i in test_block_combo])

        # Train indices: all samples NOT in test blocks
        train_block_ids = [i for i in range(n_splits) if i not in test_block_combo]
        train_indices   = np.concatenate([blocks[i] for i in train_block_ids])

        # Step 3: Apply embargo — remove train samples adjacent to test boundaries
        purge_set = set()
        for block_idx in test_block_combo:
            block = blocks[block_idx]
            # Purge before block start
            for i in range(block[0] - embargo_size, block[0]):
                if i >= 0:
                    purge_set.add(i)
            # Purge after block end
            for i in range(block[-1] + 1, block[-1] + 1 + embargo_size):
                if i < n_samples:
                    purge_set.add(i)

        # Final train set: exclude embargoed samples
        train_indices = np.array([i for i in train_indices if i not in purge_set])

        splits.append((train_indices, test_indices))

    return splits


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CPCV Backtest Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_cpcv_backtest(returns_series,
                       n_splits: int = 6,
                       n_test_splits: int = 2,
                       embargo_pct: float = 0.01) -> dict:
    """
    Run CPCV validation on a strategy's return series.

    For each CPCV path (train+test split):
        - IS Sharpe  = annualized Sharpe on training set
        - OOS Sharpe = annualized Sharpe on test set

    Aggregated results:
        - Distribution of OOS Sharpes → tells you stability of the strategy
        - PBO = fraction of paths where IS > OOS → overfitting probability

    WHAT A GOOD RESULT LOOKS LIKE:
        - Most OOS Sharpes positive → consistent edge across all time periods
        - Mean OOS Sharpe close to IS Sharpe → no overfitting
        - PBO < 0.5 → in majority of paths, good IS = good OOS

    Parameters
    ----------
    returns_series   : pd.Series or array-like  - Strategy daily returns
    n_splits         : int                      - k (number of time blocks)
    n_test_splits    : int                      - p (number of test blocks per path)
    embargo_pct      : float                    - Embargo fraction

    Returns
    -------
    dict with:
        'oos_sharpes'       : list of float - OOS Sharpe per path
        'is_sharpes'        : list of float - IS Sharpe per path
        'mean_oos_sharpe'   : float
        'median_oos_sharpe' : float
        'n_paths'           : int - Total CPCV paths evaluated
        'pbo'               : float - Probability of Backtest Overfitting [0,1]
    """
    # Convert to numpy array
    if isinstance(returns_series, pd.Series):
        r = returns_series.dropna().values.astype(float)
    else:
        r = np.array(returns_series, dtype=float)
        r = r[~np.isnan(r)]

    n_samples = len(r)

    if n_samples < n_splits * 5:
        # Not enough data for meaningful CPCV
        return {
            'oos_sharpes':       [],
            'is_sharpes':        [],
            'mean_oos_sharpe':   0.0,
            'median_oos_sharpe': 0.0,
            'n_paths':           0,
            'pbo':               1.0,
            'error': f'Insufficient data: {n_samples} samples for {n_splits} splits.',
        }

    splits     = generate_cpcv_splits(n_samples, n_splits, n_test_splits, embargo_pct)
    is_sharpes  = []
    oos_sharpes = []

    for train_idx, test_idx in splits:
        if len(train_idx) < 5 or len(test_idx) < 5:
            continue

        is_sharpes.append(_annualized_sharpe(r[train_idx]))
        oos_sharpes.append(_annualized_sharpe(r[test_idx]))

    if not oos_sharpes:
        return {
            'oos_sharpes': [], 'is_sharpes': [], 'mean_oos_sharpe': 0.0,
            'median_oos_sharpe': 0.0, 'n_paths': 0, 'pbo': 1.0,
        }

    # PBO: fraction of paths where IS Sharpe exceeds OOS Sharpe
    pbo = float(np.mean([1 if s_is > s_oos else 0
                          for s_is, s_oos in zip(is_sharpes, oos_sharpes)]))

    return {
        'oos_sharpes':       [round(s, 4) for s in oos_sharpes],
        'is_sharpes':        [round(s, 4) for s in is_sharpes],
        'mean_oos_sharpe':   round(float(np.mean(oos_sharpes)), 4),
        'median_oos_sharpe': round(float(np.median(oos_sharpes)), 4),
        'n_paths':           len(oos_sharpes),
        'pbo':               round(pbo, 4),
    }


def _annualized_sharpe(r: np.ndarray, periods: int = 252) -> float:
    """Internal helper: annualized Sharpe of a return array."""
    if len(r) < 2:
        return 0.0
    std = np.std(r, ddof=1)
    if std < 1e-10:
        return 0.0
    return float((np.mean(r) / std) * np.sqrt(periods))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Deflated Sharpe Ratio (DSR)
# ══════════════════════════════════════════════════════════════════════════════

def compute_dsr(observed_sharpe: float,
                num_trials: int,
                returns_series) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    THE PROBLEM DSR SOLVES:
        If you test 15 strategy variants on the same data, the BEST one will
        look great — but mostly due to selection bias. Standard Sharpe ratio
        doesn't account for this. DSR does.

    HOW DSR WORKS:
        1. Compute the Probabilistic Sharpe Ratio (PSR) = P(SR > SR_benchmark).
           PSR accounts for: (a) sample size T, (b) return skewness, (c) kurtosis.

        2. The benchmark is NOT zero — it's the EXPECTED MAXIMUM SR under N trials
           (what a data-miner would achieve by chance with N attempts).

        3. DSR = PSR(expected_max_SR_from_N_trials)
           A DSR > 0.95 means: "Even accounting for N trials, this strategy's
           Sharpe is real with 95% confidence."

    FORMULA:
        Step 1: Per-period SR = annualized_SR / sqrt(252)
        Step 2: SE(SR) = sqrt[ (1/T) * (1 - γ₃*SR + ((γ₄+2)/4)*SR²) ]
            where γ₃ = skewness, γ₄ = excess kurtosis
        Step 3: SR_star (expected max) using Euler-Mascheroni correction:
            SR_star = SE * [(1-γ) * Φ⁻¹(1-1/N) + γ * Φ⁻¹(1-1/(N*e))]
        Step 4: DSR = Φ[(SR - SR_star) / SE]

    Parameters
    ----------
    observed_sharpe : float  - Observed annualized Sharpe ratio to evaluate
    num_trials      : int    - Number of strategy formulations tested
    returns_series           - pd.Series or array of strategy daily returns

    Returns
    -------
    float : DSR in [0, 1].
        > 0.95  → statistically significant edge (safe to paper trade)
        0.80–0.95 → marginal; tighten risk controls
        < 0.50  → likely overfit; reject strategy
    """
    if isinstance(returns_series, pd.Series):
        r = returns_series.dropna().values.astype(float)
    else:
        r = np.array(returns_series, dtype=float)
        r = r[~np.isnan(r)]

    T = len(r)
    if T < 10:
        return 0.0

    # Convert annualized SR to per-period (daily) SR for formula consistency
    sr_hat = observed_sharpe / np.sqrt(252)

    # Higher-order moments of the return distribution
    skew = float(scipy.stats.skew(r))
    kurt = float(scipy.stats.kurtosis(r))  # Excess kurtosis (normal = 0)

    # Variance of the Sharpe ratio estimator
    # Christie (2005), Lo (2002): Var(SR_hat) ≈ (1/T)(1 - γ₃*SR + (γ₄+2)/4 * SR²)
    var_sr = (1.0 / T) * (1.0 - skew * sr_hat + ((kurt + 2.0) / 4.0) * sr_hat ** 2)
    var_sr = max(var_sr, 1.0 / T)   # Floor at 1/T to prevent negative variance
    se_sr  = np.sqrt(var_sr)

    # Expected maximum SR from N independent trials
    # Euler-Mascheroni constant (γ ≈ 0.5772) for finite-sample correction
    euler_gamma = 0.5772156649015328

    if num_trials > 1:
        # Generalized extreme value distribution expectation
        sr_star = se_sr * (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / num_trials) +
            euler_gamma         * norm.ppf(1.0 - 1.0 / (num_trials * np.e))
        )
    else:
        sr_star = 0.0   # Only 1 trial → no multiple testing correction

    # DSR = standard normal CDF of the normalized gap
    if se_sr < 1e-10:
        return 1.0 if sr_hat > sr_star else 0.0

    dsr = float(norm.cdf((sr_hat - sr_star) / se_sr))
    return round(float(np.clip(dsr, 0.0, 1.0)), 6)


def compute_prob_dsr(dsr_value: float) -> float:
    """
    P(True Edge Exists) given a DSR value.

    DSR IS already the probability that the observed Sharpe exceeds the
    expected maximum achievable by pure data-mining across N trials.
    This function simply returns it cleanly for the dashboard display.

    Returns
    -------
    float : Probability in [0, 1] that this strategy has genuine alpha.
    """
    return round(float(np.clip(dsr_value, 0.0, 1.0)), 6)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Probability of Backtest Overfitting (PBO) Utilities
# ══════════════════════════════════════════════════════════════════════════════

def interpret_pbo(pbo: float) -> str:
    """
    Human-readable interpretation of the Probability of Backtest Overfitting.

    Parameters
    ----------
    pbo : float - PBO from run_cpcv_backtest()

    Returns
    -------
    str : Interpretation string for dashboard display.
    """
    if pbo < 0.20:
        return "✅ LOW OVERFITTING — IS performance reliably predicts OOS."
    elif pbo < 0.40:
        return "✅ ACCEPTABLE — Minor IS/OOS degradation, strategy appears robust."
    elif pbo < 0.60:
        return "⚠️  MODERATE — IS significantly outperforms OOS. Tighten parameters."
    else:
        return "❌ HIGH OVERFITTING — Strategy likely curve-fitted. Reject or redesign."
