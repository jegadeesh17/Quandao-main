# Quandao: Reverse-Engineering Curriculum & Usage Guide

As a fresh graduate stepping into a Quant role, the most important thing to understand is **why** this system is built the way it is. Retail traders build monolithic scripts (one big Python file that downloads data, calculates an indicator, and places a trade). Institutions build **modular pipelines**.

---

## Chapter 1: System Architecture & The "Why"

Quandao strictly separates responsibilities into modules. The flow of data works like an assembly line:

1. **Ingestion (`data/load.py` & `database.py`)**: Raw prices arrive here. The code standardizes timestamps, fills missing values, and serves clean DataFrames.
2. **Alpha Generation (`data/factor_engine.py`)**: The clean DataFrame goes in, and new columns (factor scores) are appended. *No trading decisions happen here.*
3. **Cross-Sectional Scoring (`strategies/multi_factor.py`)**: We look at *all* stocks at once, neutralizing sector biases and weighing factors by their predictive power (Information Coefficient).
4. **Signal Generation (`strategies/alpha_signals.py`)**: The continuous scores are converted into discrete `LONG`, `SHORT`, or `NO_TRADE` signals.
5. **Risk Overlay (`risk/risk_manager.py`)**: The signal asks for permission to trade. Risk manager checks volatility, portfolio delta, and drawdowns. If safe, it calculates position size.
6. **Execution (`execution/options_executor.py`)**: The approved trade is mapped to a specific Option strike using Black-Scholes delta, and the dry-run order is logged.

> **Reverse Engineering Task 1:**
> Open `quandao_project/execution/options_executor.py`. Scroll down to `run_options_paper_trade()`. Read through the `for sig_item in signals:` loop. You will literally see the pipeline happening step-by-step: Risk Check → Order Build → Trade Risk Check → Log.

---

## Chapter 2: The Alpha Engine (Factor Math)

The `factor_engine.py` is the brain. It calculates 6 specific factors. Let's break down the math and the intuition for the most important ones.

### 1. Intermediate Momentum (6-minus-1 Month)
* **The Code:** `factor_intermediate_momentum()`
* **The Concept:** Stocks that went up over the last 6 months tend to keep going up. BUT, stocks that went up *last month* tend to reverse (mean-reversion). 
* **The Math:** We calculate the return from 6 months ago to 1 month ago, entirely skipping the most recent 21 trading days.

### 2. Amihud Illiquidity Ratio
* **The Code:** `factor_amihud_liquidity()`
* **The Concept:** How much does a stock's price move per ₹1 of trading volume? If a stock moves 5% on very little volume, it is highly illiquid. 
* **The Math:** Absolute daily return divided by daily Rupee volume. We then take a 21-day rolling average. In the code, we make this negative so that a HIGHER score means MORE liquid.

### 3. Quarterly Low Volatility
* **The Code:** `factor_quarterly_low_volatility()`
* **The Concept:** The "Low Vol Anomaly". According to traditional finance (CAPM), higher risk = higher reward. In reality, highly volatile stocks underperform because retail traders overpay for them like lottery tickets. We want *boring, low-volatility* stocks.
* **The Math:** Standard deviation of daily returns over 63 days (1 quarter), inverted.

> **Reverse Engineering Task 2:**
> Open `quandao_project/data/factor_engine.py`. Look at how `rolling().mean()` and `rolling().std()` are used extensively. Notice how `np.where` is used to avoid dividing by zero. Handling zeros and NaNs is 50% of a Quant Engineer's job.

---

## Chapter 3: Signal Validation & Statistics

Before risking money, Quants must mathematically prove their factors work on historical data without overfitting.

### Information Coefficient (IC)
Located in `strategies/multi_factor.py` (`compute_ic`).
IC is simply the **Spearman Rank Correlation** between your factor scores today and the stock returns tomorrow. 
- Why Spearman? It uses *ranks* instead of raw numbers, meaning it isn't ruined by outlier days where a stock jumps 20%.
- If IC = 0.05 to 0.10, you have a solid, tradable factor.

### Combinatorial Purged Cross-Validation (CPCV)
Located in `strategies/cpcv_validation.py`.
Machine Learning usually splits data into 80% Train, 20% Test. But financial data is a time-series. If you train on Monday and Wednesday to predict Tuesday, you are "looking into the future" (Data Leakage). 
- **Purging (Embargo):** CPCV deletes a small window of data (e.g., 1%) between the training and testing sets so they don't overlap.
- **Combinatorial:** It generates dozens of different train/test path combinations to see if the strategy survives in *all* market regimes, not just one.

### Deflated Sharpe Ratio (DSR)
If you test 100 random strategies, 1 of them will look like a genius strategy just by pure luck (this is called the Multiple Testing Problem).
DSR adjusts your Sharpe Ratio by taking into account *how many variations you tested*. It uses the Euler-Mascheroni constant to calculate the expected maximum Sharpe by chance, and returns the probability that your edge is real.

> **Reverse Engineering Task 3:**
> Open `cpcv_validation.py` and read the `compute_dsr()` function. Look at how it incorporates `scipy.stats.skew` and `kurtosis`. Financial returns are NOT normally distributed (they have fat tails). DSR mathematically accounts for these fat tails.

---

## Chapter 4: Options Math (Black-Scholes & Volatility)

When you trade stocks, you only care about direction. When you trade options, you care about Direction (Delta), Time (Theta), and Volatility (Vega).

### The Black-Scholes Function
Located in `strategies/options_pricing.py` (`black_scholes()`).
This function takes Spot (S), Strike (K), Time to Expiry (T), Risk-Free Rate (r), and Implied Volatility (sigma) and returns the theoretical price and all 5 Greeks.
- **Delta ($\Delta$)**: If the stock moves ₹1, the option moves by ₹Delta.
- **Theta ($\Theta$)**: How much ₹ value the option loses every single day it is held.
- **Gamma ($\Gamma$)**: How much your Delta increases as the stock moves in your favor.

### The Newton-Raphson Solver
Located in `strategies/options_pricing.py` (`iv_newton_raphson()`).
**This is heavy interview material.**
The market tells you the price of an option. But you need to know the Implied Volatility ($\sigma$) to know if the option is expensive or cheap. You cannot reverse the Black-Scholes algebra to solve for $\sigma$.
Instead, we use Calculus (Newton's Method):
1. Guess a Volatility (e.g., 20%).
2. Calculate the Black-Scholes price using 20%.
3. If our price is lower than the market price, we increase our guess.
4. We use **Vega** (the derivative of price with respect to volatility) to know exactly *how much* to adjust our guess.
5. We repeat this 5-10 times until our calculated price matches the market price exactly.

> **Reverse Engineering Task 4:**
> Read `select_strike_by_delta()`. Notice how the system doesn't say "buy the 24500 strike". It says "find me the strike that has a Delta of 0.40". This makes the strategy robust whether the market is at 10,000 or 25,000.

---

## Chapter 5: Risk & Institutional Execution

A beautiful strategy is useless without rigorous execution modeling. 

### GARCH Volatility Regime Detection
Located in `strategies/alpha_signals.py` (`estimate_garch_volatility()`).
Markets alternate between calm periods and violent, choppy periods. GARCH (Generalized Autoregressive Conditional Heteroscedasticity) uses yesterday's volatility and yesterday's price shock to predict today's volatility. 
**Asymmetry:** GARCH mathematically understands that a 2% *crash* causes much more panic (and future volatility) than a 2% *rally*.

### Volatility-Scaled Sizing
Located in `risk/risk_manager.py` (`scale_position_for_volatility()`).
If GARCH predicts the market will be twice as volatile tomorrow, this function automatically cuts your position size in half. This ensures your portfolio P&L swings by the exact same dollar amount every day, regardless of market chaos.

### Transaction Cost Analysis (TCA)
Located in `execution/order_executor.py` (`run_tca_simulation()`).
Retail traders say: "I bought at 100, sold at 105. I made ₹5."
Quants decompose that ₹5:
*   **Signal Alpha**: +₹5.00
*   **Bid-Ask Slippage**: -₹0.15 (You buy at the Ask, sell at the Bid)
*   **Explicit Costs**: -₹0.08 (Brokerage, STT, Exchange fees, GST)
*   **Market Impact**: -₹0.02 (Your large order pushed the price against you)
*   **Net Realized P&L**: +₹4.75

> **Reverse Engineering Task 5:**
> Open `order_executor.py`. Look at how heavily logged everything is. In production, every single micro-decision, slippage point, and greek is saved as a JSON object so that Quants can run post-trade analysis to see where money is leaking.

---

## Chapter 6: Mock Interview Prep

If you put this project on your resume, expect these exact questions in a Quant Developer/Researcher interview:

**Q: "Why did you use CPCV instead of standard cross-validation?"**
*A: "Financial time series have high serial correlation (autocorrelation). If I use K-fold, a test sample on Tuesday will be highly correlated with a training sample on Monday, leaking future data into my training set. I implemented Combinatorial Purged CV to 'embargo' (delete) the observations around the test set boundaries, completely severing that correlation."*

**Q: "How did you solve for Implied Volatility?"**
*A: "Black-Scholes isn't algebraically invertible for sigma. I built a Newton-Raphson root-finding algorithm. Because Vega is the partial derivative of price with respect to volatility, I used Vega as the denominator in the Newton step update. It converges in about 5 to 10 iterations. For deep OTM options where Vega approaches zero, I built a Brent's Bisection fallback to prevent division by zero errors."*

**Q: "How did you handle execution modeling?"**
*A: "I built a TCA (Transaction Cost Analysis) module. I didn't just subtract a flat fee. I modeled exact NSE STT, Exchange charges, and GST. Furthermore, I implemented an Almgren-Chriss style market impact model where impact scales with the square root of (order quantity / average daily volume) multiplied by realized volatility."*

**Q: "How are you weighting your multi-factor model?"**
*A: "I compute the cross-sectional Spearman Rank Information Coefficient (IC) for each factor on a rolling 63-day window. Factors are then weighted by their ICIR (IC divided by the standard deviation of IC). If a factor's ICIR drops below a threshold, its weight goes to zero, ensuring we only allocate risk to currently predictive factors."*
