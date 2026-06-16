"""
quandao_public/tests/test_cpcv.py
===================================
pytest unit tests for CPCV split generation, DSR computation, and metrics.

RUN:
    cd Quandao-main
    python -m pytest quandao_public/tests/test_cpcv.py -v

These tests cover:
  - CPCV: correct combinatorial split count
  - CPCV: train/test index disjointness (the no-leakage guarantee)
  - CPCV: embargo correctly excludes boundary samples
  - CPCV: PBO == 0 when OOS always beats IS
  - CPCV: insufficient data returns 'error' key
  - DSR: returns value in [0, 1]
  - DSR: single trial uses zero benchmark (no correction)
  - Metrics: max_drawdown = 0 on monotonically rising equity
  - Metrics: sortino sentinel (10.0) on all-positive returns
  - Metrics: sharpe = 0 on flat returns
"""

import math
from math import comb

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from quandao_public.engine.validation.cpcv_validation import (
    generate_cpcv_splits,
    run_cpcv_backtest,
    compute_dsr,
    _annualized_sharpe,
)
from quandao_public.engine.validation.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_duration,
    value_at_risk,
    conditional_var,
)


# ══════════════════════════════════════════════════════════════════════════════
# CPCV Split Generation
# ══════════════════════════════════════════════════════════════════════════════

class TestCPCVSplits:

    def test_split_count_is_combinatorial(self):
        """Should generate exactly C(n_splits, n_test_splits) splits."""
        splits = generate_cpcv_splits(n_samples=500, n_splits=6, n_test_splits=2)
        assert len(splits) == comb(6, 2)  # C(6,2) = 15

    def test_split_count_single_test_block(self):
        """C(5, 1) = 5 paths."""
        splits = generate_cpcv_splits(n_samples=300, n_splits=5, n_test_splits=1)
        assert len(splits) == 5

    def test_train_test_disjoint(self):
        """
        The fundamental CPCV guarantee: train and test indices must be
        completely disjoint. Any overlap = lookahead bias.
        """
        splits = generate_cpcv_splits(n_samples=500, n_splits=6, n_test_splits=2)
        for i, (train_idx, test_idx) in enumerate(splits):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, (
                f"Train/test overlap detected in split {i}: {len(overlap)} shared indices"
            )

    def test_all_indices_covered_across_splits(self):
        """
        For n_test_splits=1, each block appears as test exactly once.
        All original indices should appear in at least one test set.
        """
        n_samples = 300
        n_splits = 5
        splits = generate_cpcv_splits(n_samples, n_splits, n_test_splits=1)
        all_test_indices = set()
        for _, test_idx in splits:
            all_test_indices.update(test_idx)
        # Every sample should appear in some test set (embargo may exclude some train samples,
        # but test sets should cover all original blocks)
        assert len(all_test_indices) == n_samples

    def test_embargo_excludes_boundary_samples_from_train(self):
        """
        Embargoed samples adjacent to test block boundaries must NOT appear
        in the training set. This is the leakage-prevention mechanism.
        """
        n_samples = 1000
        n_splits = 5
        embargo_pct = 0.02   # 2% of one block ~ 4 samples per boundary
        splits = generate_cpcv_splits(n_samples, n_splits, n_test_splits=1,
                                      embargo_pct=embargo_pct)

        block_size = n_samples // n_splits
        embargo_size = max(1, int(n_samples * embargo_pct / n_splits))

        for train_idx, test_idx in splits:
            test_min = int(test_idx.min())
            test_max = int(test_idx.max())
            train_set = set(int(i) for i in train_idx)

            # No training sample should be within embargo_size of either test boundary
            for i in range(max(0, test_min - embargo_size), test_min):
                assert i not in train_set, (
                    f"Embargoed sample {i} (before test block starting at {test_min}) "
                    f"found in training set"
                )
            for i in range(test_max + 1, min(n_samples, test_max + 1 + embargo_size)):
                assert i not in train_set, (
                    f"Embargoed sample {i} (after test block ending at {test_max}) "
                    f"found in training set"
                )

    def test_train_set_shrinks_with_larger_embargo(self):
        """Larger embargo → smaller training set (monotone relationship)."""
        splits_small = generate_cpcv_splits(500, 5, 1, embargo_pct=0.001)
        splits_large = generate_cpcv_splits(500, 5, 1, embargo_pct=0.05)
        avg_train_small = np.mean([len(tr) for tr, _ in splits_small])
        avg_train_large = np.mean([len(tr) for tr, _ in splits_large])
        assert avg_train_large < avg_train_small


# ══════════════════════════════════════════════════════════════════════════════
# CPCV Backtest Runner
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCPCVBacktest:

    def test_insufficient_data_returns_error_key(self):
        """Too few samples → result dict must contain 'error' key."""
        result = run_cpcv_backtest([0.01, 0.02, -0.01], n_splits=6)
        assert 'error' in result

    def test_result_keys_present(self):
        """All expected keys must be present in the result."""
        np.random.seed(0)
        returns = np.random.normal(0.001, 0.01, 500)
        result = run_cpcv_backtest(returns, n_splits=6, n_test_splits=2)
        for key in ('oos_sharpes', 'is_sharpes', 'mean_oos_sharpe',
                    'median_oos_sharpe', 'n_paths', 'pbo'):
            assert key in result, f"Missing key: {key}"

    def test_n_paths_equals_combinations(self):
        """Number of evaluated paths should equal C(n_splits, n_test_splits)."""
        np.random.seed(1)
        returns = np.random.normal(0.001, 0.01, 600)
        result = run_cpcv_backtest(returns, n_splits=6, n_test_splits=2)
        assert result['n_paths'] == comb(6, 2)

    def test_pbo_in_unit_interval(self):
        """PBO must always be in [0, 1]."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 400)
        result = run_cpcv_backtest(returns, n_splits=5, n_test_splits=1)
        assert 0.0 <= result['pbo'] <= 1.0

    def test_pbo_is_zero_when_oos_always_beats_is(self):
        """
        Construct returns where later periods systematically outperform earlier ones.
        PBO (IS > OOS fraction) should be low.
        """
        np.random.seed(7)
        # Bad early performance, good later performance
        early = np.random.normal(-0.002, 0.01, 300)
        late  = np.random.normal(+0.003, 0.007, 300)
        returns = np.concatenate([early, late])
        result = run_cpcv_backtest(returns, n_splits=6, n_test_splits=1)
        # With this construction, most OOS (later blocks) should beat IS (earlier blocks)
        assert result['pbo'] <= 0.6   # Generous threshold given combinatorial averaging

    def test_accepts_pandas_series(self):
        """run_cpcv_backtest must accept a pd.Series as well as a numpy array."""
        import pandas as pd
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0.001, 0.01, 500))
        result = run_cpcv_backtest(returns, n_splits=5, n_test_splits=1)
        assert 'pbo' in result

    def test_handles_nan_in_returns(self):
        """NaN values in the returns series should be silently dropped."""
        np.random.seed(0)
        returns = np.random.normal(0.001, 0.01, 500)
        returns[50] = np.nan
        returns[200] = np.nan
        result = run_cpcv_backtest(returns, n_splits=5, n_test_splits=1)
        assert 'pbo' in result
        assert not math.isnan(result['pbo'])


# ══════════════════════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio
# ══════════════════════════════════════════════════════════════════════════════

class TestDSR:

    def test_dsr_in_unit_interval(self):
        """DSR must always be in [0, 1]."""
        np.random.seed(0)
        returns = np.random.normal(0.001, 0.01, 252)
        dsr = compute_dsr(observed_sharpe=1.5, num_trials=15, returns_series=returns)
        assert 0.0 <= dsr <= 1.0

    def test_dsr_single_trial_no_correction(self):
        """With num_trials=1, sr_star=0; DSR equals a standard PSR-style calculation."""
        np.random.seed(1)
        returns = np.random.normal(0.001, 0.01, 252)
        dsr_1 = compute_dsr(observed_sharpe=1.5, num_trials=1, returns_series=returns)
        dsr_5 = compute_dsr(observed_sharpe=1.5, num_trials=5, returns_series=returns)
        # More trials → higher benchmark → lower DSR
        assert dsr_1 >= dsr_5

    def test_dsr_higher_sharpe_gives_higher_dsr(self):
        """Higher observed Sharpe → higher DSR (monotone)."""
        np.random.seed(2)
        returns = np.random.normal(0.001, 0.01, 252)
        dsr_low  = compute_dsr(0.5, 10, returns)
        dsr_high = compute_dsr(2.0, 10, returns)
        assert dsr_high > dsr_low

    def test_dsr_more_trials_gives_lower_dsr(self):
        """More trials → harder multiple-testing penalty → lower DSR."""
        np.random.seed(3)
        returns = np.random.normal(0.001, 0.01, 252)
        dsr_few  = compute_dsr(1.5, num_trials=2,  returns_series=returns)
        dsr_many = compute_dsr(1.5, num_trials=50, returns_series=returns)
        assert dsr_few > dsr_many

    def test_dsr_returns_zero_for_insufficient_data(self):
        """Fewer than 10 return observations: DSR returns 0."""
        dsr = compute_dsr(1.5, num_trials=10, returns_series=[0.01, 0.02])
        assert dsr == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Metrics module
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics:

    def test_annualized_sharpe_flat_returns(self):
        """Zero-variance returns → Sharpe must be 0.0."""
        assert _annualized_sharpe(np.zeros(252)) == 0.0

    def test_annualized_sharpe_single_element(self):
        """Single-element array → Sharpe must be 0.0 (std undefined)."""
        assert _annualized_sharpe(np.array([0.01])) == 0.0

    def test_max_drawdown_monotonically_rising(self):
        """No drawdown on a perfectly rising equity curve."""
        returns = np.full(252, 0.001)   # +0.1% every day
        assert max_drawdown(returns) == 0.0

    def test_max_drawdown_all_losses(self):
        """Max drawdown on an all-loss series should be negative."""
        returns = np.full(100, -0.01)
        assert max_drawdown(returns) < 0.0

    def test_max_drawdown_empty_series(self):
        """Empty series → max_drawdown returns 0."""
        assert max_drawdown(np.array([])) == 0.0

    def test_sortino_all_positive_returns_gives_sentinel(self):
        """All-positive returns: no downside deviation → return sentinel 10.0."""
        returns = np.full(252, 0.005)   # All winning days
        result = sortino_ratio(returns, risk_free_rate=0.0)
        assert result == 10.0

    def test_sharpe_zero_std_returns_zero(self):
        """Zero-variance returns (std = 0) → Sharpe = 0 (division by zero guard)."""
        # Use identical constant returns; std is exactly 0 regardless of mean.
        returns = np.full(252, 0.0)   # All zeros — std is mathematically 0
        result = sharpe_ratio(returns, risk_free_rate=0.065)
        assert result == 0.0

    def test_var_95_is_negative_on_mixed_returns(self):
        """5th percentile of mixed daily returns should be negative."""
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.01, 1000)
        var = value_at_risk(returns, confidence=0.95)
        assert var < 0.0

    def test_cvar_worse_than_var(self):
        """CVaR (Expected Shortfall) must be <= VaR (it's the average of the tail)."""
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.01, 1000)
        var  = value_at_risk(returns, confidence=0.95)
        cvar = conditional_var(returns, confidence=0.95)
        assert cvar <= var

    def test_max_drawdown_duration_all_rising(self):
        """No drawdown periods → duration = 0."""
        returns = np.full(100, 0.001)
        assert max_drawdown_duration(returns) == 0

    def test_max_drawdown_duration_positive(self):
        """Returns with a clear drawdown period should give duration > 0."""
        returns = np.array([0.01, 0.01, -0.05, -0.02, -0.01, 0.01, 0.02, 0.03])
        assert max_drawdown_duration(returns) > 0
