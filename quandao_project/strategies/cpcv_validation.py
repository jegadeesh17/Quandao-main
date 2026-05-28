import numpy as np
import pandas as pd
from itertools import combinations
import scipy.stats as stats

def get_cpcv_paths(n_splits: int, n_test_splits: int) -> list:
    """
    Generate Combinatorial Purged Cross-Validation paths.
    Returns a list of dicts with 'train' and 'test' indices for the splits.
    """
    splits = list(range(n_splits))
    test_combinations = list(combinations(splits, n_test_splits))
    
    paths = []
    for test_splits in test_combinations:
        train_splits = [s for s in splits if s not in test_splits]
        paths.append({
            'train': train_splits,
            'test': list(test_splits)
        })
    return paths

def compute_dsr(sharpe_ratio: float, num_trials: int, returns_series: pd.Series = None, 
                skewness: float = 0.0, kurtosis: float = 3.0) -> float:
    """
    Computes the Deflated Sharpe Ratio (DSR).
    Adjusts for overfitting based on the number of trials tested.
    """
    if returns_series is not None:
        skewness = returns_series.skew()
        kurtosis = returns_series.kurtosis() + 3.0 # Pandas kurtosis is excess, we want raw

    # Expected max Sharpe ratio from independent trials approximation
    if num_trials > 1:
        max_sr_expected = np.sqrt(2 * np.log(num_trials))
    else:
        max_sr_expected = 0.0
    
    # Variance of the Sharpe Ratio estimation (Bailey & Lopez de Prado)
    # sr_var = 1 - (skewness * sharpe_ratio) + ((kurtosis - 1) / 4) * (sharpe_ratio ** 2)
    # To avoid negative variance for extreme skew/kurtosis, using max(0.1, var)
    var_term = 1 - (skewness * sharpe_ratio) + ((kurtosis - 1) / 4) * (sharpe_ratio ** 2)
    sr_var = max(0.1, var_term)
    
    # DSR Calculation
    adjusted_sr = (sharpe_ratio - max_sr_expected) / np.sqrt(sr_var)
    return float(adjusted_sr)

def compute_prob_dsr(dsr_val: float) -> float:
    """
    Converts a Deflated Sharpe Ratio into a Probability.
    """
    return float(stats.norm.cdf(dsr_val))

def run_cpcv_backtest(returns: pd.Series, n_splits: int = 6, n_test_splits: int = 2) -> dict:
    """
    Runs CPCV over a return series, outputting the distribution of Out-of-Sample (OOS) Sharpes.
    """
    n = len(returns)
    split_size = n // n_splits
    paths = get_cpcv_paths(n_splits, n_test_splits)
    
    oos_sharpes = []
    is_sharpes = []
    
    for path in paths:
        train_idx = []
        test_idx = []
        
        for i in range(n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < n_splits - 1 else n
            
            if i in path['train']:
                train_idx.extend(list(range(start, end)))
            else:
                test_idx.extend(list(range(start, end)))
                
        train_ret = returns.iloc[train_idx]
        test_ret = returns.iloc[test_idx]
        
        if train_ret.std() > 0:
            is_sr = (train_ret.mean() / train_ret.std()) * np.sqrt(12)
        else:
            is_sr = 0.0
            
        if test_ret.std() > 0:
            oos_sr = (test_ret.mean() / test_ret.std()) * np.sqrt(12)
        else:
            oos_sr = 0.0
            
        is_sharpes.append(is_sr)
        oos_sharpes.append(oos_sr)
        
    return {
        "is_sharpes": is_sharpes,
        "oos_sharpes": oos_sharpes,
        "mean_oos_sharpe": np.mean(oos_sharpes),
        "median_oos_sharpe": np.median(oos_sharpes)
    }
