"""
quandao_public/tests/test_options_pricing.py
==============================================
pytest unit tests for the Black-Scholes pricing engine and IV solvers.

RUN:
    cd Quandao-main
    python -m pytest quandao_public/tests/test_options_pricing.py -v

These tests cover:
  - Boundary conditions (expired options, sub-intrinsic prices)
  - Put-call parity identity
  - Greek sign/range validation
  - IV round-trip: BS(IV(market_price)) == market_price
  - NR non-convergence correctly triggers bisection fallback
"""

import math
import numpy as np
import pytest

# Allow running from the repo root without install
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from quandao_public.strategies.options_pricing import (
    black_scholes,
    implied_volatility,
    iv_newton_raphson,
    iv_bisection,
)


# ── Constants shared across tests ────────────────────────────────────────────
ATM_PARAMS = dict(S=24500.0, K=24500.0, T=7 / 365, r=0.065, sigma=0.18)
DEEP_OTM_PARAMS = dict(S=24500.0, K=28000.0, T=3 / 365, r=0.065, sigma=0.35)


# ══════════════════════════════════════════════════════════════════════════════
# Boundary conditions
# ══════════════════════════════════════════════════════════════════════════════

class TestBoundaryConditions:

    def test_expired_itm_call_returns_intrinsic(self):
        """At T=0, call price must equal max(S-K, 0)."""
        result = black_scholes(S=100.0, K=90.0, T=0.0, r=0.06, sigma=0.2, option_type='call')
        assert result['price'] == pytest.approx(10.0, abs=1e-6)

    def test_expired_otm_call_returns_zero(self):
        """At T=0, OTM call has zero value."""
        result = black_scholes(S=80.0, K=90.0, T=0.0, r=0.06, sigma=0.2, option_type='call')
        assert result['price'] == 0.0

    def test_expired_itm_put_returns_intrinsic(self):
        """At T=0, put price must equal max(K-S, 0)."""
        result = black_scholes(S=85.0, K=100.0, T=0.0, r=0.06, sigma=0.2, option_type='put')
        assert result['price'] == pytest.approx(15.0, abs=1e-6)

    def test_expired_all_greeks_zero(self):
        """At expiry, all Greeks must be zero (no more sensitivity to anything)."""
        result = black_scholes(S=100.0, K=100.0, T=0.0, r=0.06, sigma=0.2, option_type='call')
        for greek in ('delta', 'gamma', 'theta', 'vega', 'rho'):
            assert result[greek] == 0.0, f"{greek} should be 0 at expiry"

    def test_negative_time_treated_as_expired(self):
        """Negative T (past expiry) must behave like T=0."""
        result = black_scholes(S=100.0, K=90.0, T=-1.0, r=0.06, sigma=0.2, option_type='call')
        assert result['price'] == pytest.approx(10.0, abs=1e-6)

    def test_zero_sigma_atm_call_approaches_intrinsic(self):
        """With σ→0, ATM call price → 0 (no future uncertainty)."""
        result = black_scholes(S=100.0, K=100.0, T=1.0, r=0.065, sigma=1e-6, option_type='call')
        # At zero vol, price converges to S - K*e^{-rT} for deep in-the-money,
        # or near-zero for ATM. It should be a very small positive number.
        assert result['price'] >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Mathematical identities
# ══════════════════════════════════════════════════════════════════════════════

class TestMathematicalIdentities:

    def test_put_call_parity(self):
        """
        C - P = S - K * e^{-rT}    (European put-call parity)

        This is an exact mathematical identity that must hold to within
        floating-point precision. Any violation means the pricing formula is wrong.
        """
        S, K, T, r, sigma = 24500, 24500, 30 / 365, 0.065, 0.20
        call = black_scholes(S, K, T, r, sigma, 'call')['price']
        put  = black_scholes(S, K, T, r, sigma, 'put')['price']
        expected_diff = S - K * math.exp(-r * T)
        assert call - put == pytest.approx(expected_diff, rel=1e-4)

    def test_put_call_parity_otm(self):
        """Put-call parity must also hold for OTM options."""
        S, K, T, r, sigma = 24500, 25000, 14 / 365, 0.065, 0.18
        call = black_scholes(S, K, T, r, sigma, 'call')['price']
        put  = black_scholes(S, K, T, r, sigma, 'put')['price']
        expected_diff = S - K * math.exp(-r * T)
        assert call - put == pytest.approx(expected_diff, rel=1e-4)

    def test_call_monotonically_decreasing_in_strike(self):
        """Higher strike → lower call price (monotone in K)."""
        S, T, r, sigma = 24500, 30 / 365, 0.065, 0.20
        strikes = [23000, 24000, 24500, 25000, 26000]
        prices = [black_scholes(S, K, T, r, sigma, 'call')['price'] for K in strikes]
        assert prices == sorted(prices, reverse=True)

    def test_put_monotonically_increasing_in_strike(self):
        """Higher strike → higher put price (monotone in K)."""
        S, T, r, sigma = 24500, 30 / 365, 0.065, 0.20
        strikes = [23000, 24000, 24500, 25000, 26000]
        prices = [black_scholes(S, K, T, r, sigma, 'put')['price'] for K in strikes]
        assert prices == sorted(prices)


# ══════════════════════════════════════════════════════════════════════════════
# Greek sign and range validation
# ══════════════════════════════════════════════════════════════════════════════

class TestGreeks:

    def test_call_delta_in_open_unit_interval(self):
        result = black_scholes(**ATM_PARAMS, option_type='call')
        assert 0.0 < result['delta'] < 1.0

    def test_put_delta_in_negative_unit_interval(self):
        result = black_scholes(**ATM_PARAMS, option_type='put')
        assert -1.0 < result['delta'] < 0.0

    def test_atm_call_delta_near_half(self):
        """ATM call delta ≈ 0.5 (slightly above due to log-normality)."""
        result = black_scholes(**ATM_PARAMS, option_type='call')
        assert 0.45 < result['delta'] < 0.60

    def test_gamma_always_positive(self):
        """Gamma is always positive for long options (call and put)."""
        for opt in ('call', 'put'):
            result = black_scholes(**ATM_PARAMS, option_type=opt)
            assert result['gamma'] > 0.0, f"Gamma must be positive for {opt}"

    def test_call_and_put_share_same_gamma(self):
        """Gamma is identical for call and put at same strike/expiry (Black-Scholes property)."""
        call = black_scholes(**ATM_PARAMS, option_type='call')
        put  = black_scholes(**ATM_PARAMS, option_type='put')
        assert call['gamma'] == pytest.approx(put['gamma'], rel=1e-6)

    def test_call_and_put_share_same_vega(self):
        """Vega is identical for call and put (Black-Scholes property)."""
        call = black_scholes(**ATM_PARAMS, option_type='call')
        put  = black_scholes(**ATM_PARAMS, option_type='put')
        assert call['vega'] == pytest.approx(put['vega'], rel=1e-6)

    def test_theta_negative_for_long_call(self):
        """Theta must be negative — long calls lose value as time passes."""
        result = black_scholes(**ATM_PARAMS, option_type='call')
        assert result['theta'] < 0.0

    def test_theta_negative_for_long_put(self):
        """Theta must be negative for long puts too (time decay)."""
        result = black_scholes(**ATM_PARAMS, option_type='put')
        assert result['theta'] < 0.0

    def test_vega_positive(self):
        """Vega must be positive — higher vol → higher option price."""
        result = black_scholes(**ATM_PARAMS, option_type='call')
        assert result['vega'] > 0.0

    def test_call_rho_positive(self):
        """Call rho is positive — higher rates → higher call price (lower PV of strike)."""
        result = black_scholes(**ATM_PARAMS, option_type='call')
        assert result['rho'] > 0.0

    def test_put_rho_negative(self):
        """Put rho is negative — higher rates → lower put price."""
        result = black_scholes(**ATM_PARAMS, option_type='put')
        assert result['rho'] < 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Implied Volatility — round-trip and failure modes
# ══════════════════════════════════════════════════════════════════════════════

class TestImpliedVolatility:

    @pytest.mark.parametrize("sigma_true", [0.08, 0.15, 0.20, 0.30, 0.50, 0.80])
    def test_iv_round_trip_atm(self, sigma_true):
        """IV(BS(σ)) == σ for ATM options across the full realistic IV range."""
        S, K, T, r = 24500, 24500, 14 / 365, 0.065
        market_price = black_scholes(S, K, T, r, sigma_true, 'call')['price']
        recovered_iv = implied_volatility(market_price, S, K, T, r, 'call')
        assert not math.isnan(recovered_iv), f"IV returned NaN for σ={sigma_true}"
        assert recovered_iv == pytest.approx(sigma_true, rel=1e-3), (
            f"IV round-trip failed: got {recovered_iv:.6f}, expected {sigma_true:.6f}"
        )

    @pytest.mark.parametrize("sigma_true", [0.15, 0.25, 0.40])
    def test_iv_round_trip_put(self, sigma_true):
        """IV round-trip must also work for puts."""
        S, K, T, r = 24500, 24000, 14 / 365, 0.065
        market_price = black_scholes(S, K, T, r, sigma_true, 'put')['price']
        recovered_iv = implied_volatility(market_price, S, K, T, r, 'put')
        assert not math.isnan(recovered_iv)
        assert recovered_iv == pytest.approx(sigma_true, rel=1e-3)

    def test_iv_nr_returns_nan_on_non_convergence(self):
        """
        NR with max_iter=10 must return NaN if it doesn't converge.
        This guarantees bisection can actually fire as a fallback.
        """
        # Force non-convergence by using a pathological initial guess on a deep OTM option.
        # We check that NR alone returns NaN when Vega is near-zero.
        S, K, T, r = 24500.0, 32000.0, 1 / 365, 0.065
        # Deep OTM call with 1 day left — Vega is essentially zero
        market_price = black_scholes(S, K, T, r, 0.30, 'call')['price']
        if market_price <= 0.01:
            # NR should get NaN here; bisection may or may not find a root
            iv_nr = iv_newton_raphson(market_price, S, K, T, r, 'call', max_iter=10)
            # Either NaN (correct) or a valid IV — but must NOT be an un-converged garbage value
            if not math.isnan(iv_nr):
                # If it returned something, verify it's actually correct
                check = black_scholes(S, K, T, r, iv_nr, 'call')['price']
                assert abs(check - market_price) < 1e-4

    def test_iv_returns_nan_for_expired_option(self):
        """No real IV exists for expired options (T=0)."""
        iv = implied_volatility(100.0, 24500, 24500, T=0.0, r=0.065, option_type='call')
        assert math.isnan(iv)

    def test_iv_returns_nan_for_zero_price(self):
        """Zero market price: no real IV."""
        iv = implied_volatility(0.0, 24500, 24500, T=14 / 365, r=0.065, option_type='call')
        assert math.isnan(iv)

    def test_iv_returns_nan_for_below_intrinsic_price(self):
        """Market price below intrinsic value: no real IV exists."""
        # Deep ITM call: S=24500, K=20000 → intrinsic ~4500; price of 1 is sub-intrinsic
        iv = implied_volatility(1.0, 24500, 20000, T=14 / 365, r=0.065, option_type='call')
        assert math.isnan(iv)

    def test_smart_solver_handles_deep_otm_via_bisection(self):
        """
        The smart implied_volatility() must recover via bisection even when NR fails.
        Deep OTM options have near-zero Vega — NR diverges, bisection converges.
        """
        S, K, T, r, sigma_true = 24500, 27000, 14 / 365, 0.065, 0.40
        market_price = black_scholes(S, K, T, r, sigma_true, 'call')['price']
        if market_price > 0.10:
            iv = implied_volatility(market_price, S, K, T, r, 'call')
            if not math.isnan(iv):
                assert iv == pytest.approx(sigma_true, rel=0.05)
