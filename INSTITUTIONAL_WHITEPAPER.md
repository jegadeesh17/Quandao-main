# Quandao: Institutional Quantitative Trading Architecture Whitepaper

*This document serves as the rigorous, PhD-level technical documentation for the Quandao Quantitative Architecture. It maps the theoretical mathematical constructs directly to their vectorized Python implementations, explicitly detailing edge-case handling, micro-structure friction modeling, and statistical validation proofs.*

---

## 1. Multi-Factor Alpha Generation Model
The engine relies on a cross-sectional multi-factor model designed to capture risk premia while isolating idiosyncratic alpha through sector-neutralization. 

### 1.1 Amihud Illiquidity Premium (`factor_amihud_liquidity`)
The Amihud (2002) illiquidity measure captures the price impact of trading volume.
**Mathematical Definition:**
$$ ILLIQ_{i,t} = \frac{1}{D_{i,t}} \sum_{d=1}^{D_{i,t}} \frac{|R_{i,v,d}|}{V_{i,v,d} \times P_{i,v,d}} $$
Where $R$ is return, $V$ is volume, and $P$ is price. 

**Implementation Details (`data/factor_engine.py`):**
*   **Vectorization**: The algorithm computes daily absolute returns divided by Rupee Volume (`close * volume`). 
*   **Edge Case Handling**: Micro-cap stocks frequently exhibit zero-volume days, leading to `ZeroDivisionError` (or `inf` in Pandas). The implementation utilizes `np.where(volume == 0, np.nan, ...)` to mask zero-volume days prior to calculating the 21-day rolling mean, preventing cascading NaN propagation in cross-sectional scoring.
*   **Inversion**: The raw Amihud score is inverted (multiplied by -1) to align the factor directionally (higher score = higher liquidity).

### 1.2 Information Coefficient (IC) & ICIR Weighting (`strategies/multi_factor.py`)
Rather than static equal-weighting, the engine dynamically allocates factor weights based on predictive efficacy.
**Mathematical Definition:**
$$ IC_t = \rho_{spearman}(\text{Factor}_{t}, \text{Return}_{t+1}) $$
$$ ICIR = \frac{\overline{IC}}{\sigma_{IC}} $$

**Implementation Details:**
*   `compute_ic()` calculates the cross-sectional Spearman Rank correlation between the $t_0$ z-scored factor and the $t_{+1}$ forward return. Spearman is strictly enforced over Pearson to mitigate the non-normal, leptokurtic distribution of financial returns.
*   In `generate_multi_factor_composite()`, rolling 63-day ICIR is computed. If the ICIR drops below statistical significance (e.g., $< 0.10$), the factor's cross-sectional weight decays to zero, acting as an automated factor-timing mechanism.

---

## 2. Statistical Validation Framework

Standard Machine Learning Cross-Validation (K-Fold) assumes I.I.D (Independent and Identically Distributed) data. Financial time series exhibit heavy serial correlation (autocorrelation). Using K-Fold results in look-ahead bias and data leakage. 

### 2.1 Combinatorial Purged Cross-Validation (CPCV) (`strategies/cpcv_validation.py`)
**Implementation:**
The `generate_cpcv_splits()` function divides the dataset into $N$ groups and selects $K$ groups for testing. 
*   **Purging**: The function calculates a 1% `embargo` index threshold. It explicitly drops $N$ indices immediately preceding and succeeding the test boundary. This severs the autoregressive correlation (e.g., an AR(1) process where $P_t$ is highly correlated to $P_{t-1}$).

### 2.2 Deflated Sharpe Ratio (DSR) & Probability of Backtest Overfitting (PBO)
A strategy generating a Sharpe of 2.0 is meaningless if 1,000 variations were tested (Multiple Testing Problem). 
**Mathematical Definition (Bailey & López de Prado, 2014):**
$$ DSR = \Phi \left( \frac{(SR_{test} - SR_{expected})\sqrt{T-1}}{\sqrt{1 - \gamma_3 SR_{test} + \frac{\gamma_4 - 1}{4} SR_{test}^2}} \right) $$
Where $SR_{expected}$ leverages the Euler-Mascheroni constant ($\gamma \approx 0.5772$) to estimate the expected maximum Sharpe ratio of $N$ independent trials:
$$ SR_{expected} = \sqrt{2 \ln(N)} + \frac{\gamma}{\sqrt{2 \ln(N)}} $$

**Implementation Details (`cpcv_validation.py`):**
*   `compute_dsr()` strictly calculates the third moment (Skewness, $\gamma_3$) and fourth moment (Kurtosis, $\gamma_4$) using `scipy.stats`. 
*   It adjusts the Sharpe ratio variance denominator to account for the heavy tails of the return distribution. It outputs a cumulative distribution function ($\Phi$) value, effectively acting as a $p$-value for the validity of the Alpha.

---

## 3. Stochastic Volatility & Derivatives Pricing

### 3.1 EGARCH(1,1) Volatility Regime Modeling (`strategies/alpha_signals.py`)
Standard realized volatility evenly weights all past observations. We implement an Exponential GARCH model to capture the "Leverage Effect" (negative returns increase volatility more than positive returns of the same magnitude).
**Implementation Details:**
*   The `estimate_garch_volatility()` function utilizes `arch_model(vol='EGARCH', p=1, o=1, q=1)`.
*   The output provides a conditional volatility forecast $h_{t+1}$. The `risk_manager.py` dynamically scales portfolio position sizes inversely to this forecast ($Size_t = \frac{TargetVol}{h_{t+1}}$).

### 3.2 Implied Volatility Newton-Raphson Solver (`strategies/options_pricing.py`)
The Black-Scholes PDE does not have a closed-form algebraic inverse for $\sigma$ (Implied Volatility).
**Mathematical Definition:**
$$ \sigma_{n+1} = \sigma_n - \frac{BS(\sigma_n) - MarketPrice}{\mathcal{V}(\sigma_n)} $$
Where Vega ($\mathcal{V}$) is the partial derivative of the option price with respect to volatility: $\frac{\partial V}{\partial \sigma}$.

**Implementation Details (`iv_newton_raphson`):**
*   **Vectorization**: The algorithm sets an initial heuristic guess ($\sigma_0 = 0.20$). It loops, computing the Black-Scholes theoretical price and Vega at $\sigma_n$.
*   **Convergence**: It subtracts the error divided by Vega. Convergence is achieved when $|BS(\sigma_n) - MarketPrice| < 10^{-4}$.
*   **Fail-Safe (Brent's Method)**: Deep Out-of-the-Money (OTM) options have Vega approaching exactly zero. The Newton-Raphson denominator $\mathcal{V}(\sigma_n)$ causes a `ZeroDivisionError` or infinite step size. The algorithm catches this instability and falls back to a Bisection root-finding method.

---

## 4. Execution Microstructure (TCA)

### 4.1 Transaction Cost Analysis (`execution/order_executor.py`)
A simulated execution engine must deduct realistic market frictions to separate Gross Alpha from Net Alpha.
**Implementation Details (`run_tca_simulation`):**
1.  **Bid-Ask Bounce**: Models crossing the spread. $Slippage = (Ask - Mid) \times Quantity$.
2.  **Explicit Frictions**: Calls `cost_model.py` to calculate exact non-linear NSE fee structures: Brokerage (capped at ₹20), STT (0.0125% for futures), Exchange Transaction Charges (0.0019%), SEBI turnover fees, and 18% GST on brokerage+exchange fees.
3.  **Almgren-Chriss Market Impact**: Models temporary market impact proportional to the square root of the normalized order size multiplied by realized volatility.
