# Quandao: Production-Grade Quantitative Trading Infrastructure

**Quandao** is an institutional-grade, low-latency, statistically rigorous algorithmic trading and research platform built for the Indian equities, futures, and options markets (NSE). 

Designed with an emphasis on **capital preservation, strict risk management, and rigorous alpha validation**, Quandao aims to bridge the gap between "toy" algorithmic projects and production-ready institutional infrastructure.

---

## 🚀 Core Architectural Features

Quandao is composed of interconnected but strictly decoupled modules, prioritizing data integrity, scalability, and testability.

### 1. Alpha Research & Factor Engine (`quandao_project/data/factor_engine.py`)
A comprehensive, purely functional factor engine calculating multi-factor composite scores based on institutional literature:
*   **Sharpe-Adjusted Trend**: Volatility-weighted momentum.
*   **Intermediate Momentum**: 6-minus-1 month return (avoids short-term reversal).
*   **Amihud Liquidity Score**: Cross-sectional liquidity filtering.
*   **Quarterly Low Volatility**: 63-day realized volatility targeting the low-vol anomaly.
*   **VPT Accumulation**: Directional volume accumulation tracker.
*   *Composite scoring uses Information Coefficient (IC) and ICIR filtering for factor weighting, combined with Sector Neutralization to isolate idiosyncratic edge.*

### 2. Rigorous Strategy Validation (`quandao_project/strategies/cpcv_validation.py`)
Standard K-Fold Cross-Validation leaks data through autocorrelation. Quandao solves this with institutional techniques:
*   **Combinatorial Purged Cross-Validation (CPCV)**: Generates combinatorial train/test paths while implementing "embargoes" around test fold boundaries to prevent temporal leakage.
*   **Deflated Sharpe Ratio (DSR)**: A probabilistic Sharpe Ratio utilizing the Euler-Mascheroni correction to penalize multiple testing bias (data-mining), returning the actual probability $P(True Edge)$.
*   **Probability of Backtest Overfitting (PBO)**: Analyzes out-of-sample (OOS) Sharpe degradation across all CPCV paths.

### 3. Quantitative Risk Management (`quandao_project/risk/risk_manager.py`)
*   **Drawdown Circuit Breakers**: Halts new entries upon exceeding the portfolio drawdown threshold (default: 10%).
*   **Volatility-Scaled Position Sizing**: Uses EGARCH(1,1) conditionally estimated volatility (`alpha_signals.py`) to inversely scale position size, ensuring constant risk exposure across changing market regimes.
*   **Kelly Criterion & Fixed Fractional Sizing**: Mathematically optimizes the growth rate of capital while enforcing maximum single-trade risk constraints.
*   **Portfolio Delta Caps**: Strict upper bounds on net directional (unhedged) portfolio delta.

### 4. Realistic Execution & Options Engine (`quandao_project/execution/`)
*   **Transaction Cost Analysis (TCA)**: Unlike naive backtesters, Quandao models the full NSE cost structure (Brokerage + STT + Exchange Charge + SEBI + Stamp Duty + GST) and estimates market impact (Almgren-Chriss model) and bid-ask slippage.
*   **Options Pricing Engine**: A standalone module implementing Black-Scholes, calculating all 5 Greeks, and providing a high-performance **Newton-Raphson Implied Volatility Solver** (with Brent's bisection fallback).
*   **Options Execution Bot**: Translates directional factor signals into delta-targeted strike selections, tracking daily theta-burn rates for dynamic exit conditions. All execution logic runs within a strict Dry-Run environment.

---

## 📂 Codebase Structure

```
Quandao-main/
│
├── quandao_project/
│   ├── app.py                      # (Optional) Lightweight Flask server
│   ├── config.py                   # Platform-wide risk, capital, and execution constants
│   ├── research_dashboard.py       # Streamlit-based Interactive Research UI
│   │
│   ├── data/                       # Unified DB connections and Data Pipelines
│   │   ├── database.py             # Single-entry access to PostgreSQL
│   │   ├── factor_engine.py        # Alpha generation pure functions
│   │   └── load.py, fetch.py       # Historical/Streaming data loading (Fyers API)
│   │
│   ├── strategies/                 # Math, Validation, and Strategy Models
│   │   ├── metrics.py              # Sharpe, Sortino, Calmar, VaR pure functions
│   │   ├── cost_model.py           # NSE realistic cost breakdowns
│   │   ├── cpcv_validation.py      # CPCV, DSR, and PBO evaluation
│   │   ├── multi_factor.py         # IC decay, Sector Neutralization, PCA Stat-Arb
│   │   ├── alpha_signals.py        # EGARCH Volatility Modeling
│   │   └── options_pricing.py      # Black-Scholes & Newton-Raphson IV Solvers
│   │
│   ├── risk/                       # Live Guardrails
│   │   └── risk_manager.py         # Circuit breakers, Vol-scaling, Delta Caps
│   │
│   ├── execution/                  # Trading Engine
│   │   ├── order_executor.py       # Dry-run execution & TCA decompositions
│   │   └── options_executor.py     # Signal-to-Strike matching & Theta-burn monitoring
│   │
│   └── results/                    # Trade logs and Post-trade analytics
│       └── paper_trades.json       # Simulated execution history
```

---

## 🛠 Getting Started

### 1. Requirements & Setup
Quandao requires Python 3.9+, PostgreSQL with the TimescaleDB extension, and access to the Fyers API (for live market data fetching).

```bash
# Clone the repository
git clone https://github.com/jegadeesh17/Quandao-main.git
cd Quandao-main

# Install dependencies (requires pip)
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/quandao
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_REDIRECT_URI=http://localhost:8080/login
```

### 3. Launching the Research Dashboard
The Streamlit research dashboard allows you to visually interact with the multi-factor screener, analyze CPCV paths, and view PCA factor models.

```bash
streamlit run quandao_project/research_dashboard.py
```

### 4. Running the Options Execution Bot
To simulate dry-run options trading logic:
```python
# Example execution pipeline within a script
from quandao_project.execution.options_executor import run_options_paper_trade

# Simulates placing orders and enforcing risk rules, writing to results/paper_trades.json
run_options_paper_trade(
    signals=[{'symbol': 'NSE:NIFTY50-INDEX', 'signal': 'LONG_CALL', 'score': 0.85}],
    spot_data={'NSE:NIFTY50-INDEX': {'spot': 24500, 'iv': 0.18, 'days_to_expiry': 7}}
)
```

---
*Disclaimer: This platform operates strictly in DRY-RUN simulated execution mode for research and portfolio demonstration. It does not place live orders to brokerages.*
