# Quandao: Institutional Quantitative Trading Architecture

---

### **Project Overview**

Financial markets exhibit complex micro-structure frictions and non-stationary volatility. This project builds Quandao, an institutional quantitative trading architecture using advanced statistical modeling and derivatives pricing to automatically identify and execute algorithmic trading strategies.

The system leverages a multi-factor alpha generation model, Combinatorial Purged Cross-Validation (CPCV), and execution microstructure (TCA) to analyze market data and predict trading signals with high confidence. The project aims to support institutional-grade research and assist algorithmic traders in robust strategy deployment.

---

### **Key Features**

* **Multi-Factor Alpha Generation:** Generates alpha using Amihud Illiquidity and dynamically weighted Information Coefficient (IC) signals.
* **Robust Validation Framework:** Implements Combinatorial Purged Cross-Validation (CPCV) to eliminate look-ahead bias and data leakage.
* **Deflated Sharpe Ratio (DSR):** Computes DSR and Probability of Backtest Overfitting (PBO) to validate strategy significance.
* **Stochastic Volatility Modeling:** Uses EGARCH(1,1) for conditional volatility forecasting and position sizing.
* **Derivatives Pricing:** Employs Newton-Raphson solver for Black-Scholes Implied Volatility calculation.
* **Interactive Flask Dashboard:** Deployed application for backtesting analysis, charting, and viewing trading signals.
* **Execution Microstructure (TCA):** Simulates realistic market frictions including slippage, exchange fees, and market impact.
* **Lazy Loading Charts:** Optimized OHLCV data fetching with pagination for responsive historical chart analysis.

---

### **Dataset**

* **Source:** NSE Historical Data (NIFTY 50, RELIANCE, HDFCBANK)
* **Coverage:** Multi-timeframe OHLCV market data
* **Data Type:** Time-series trading datasets

#### **Market Instruments**

* NIFTY 50 Index
* Reliance Industries
* HDFC Bank

#### **Key Features Analyzed**

* Market illiquidity and volume dynamics
* Cross-sectional factor correlations (Spearman Rank)
* Non-normal, leptokurtic return distributions
* Autoregressive volatility clusters
* Order execution slippage and impact

---

### **Project Structure**

```bash
quandao_public/
│
├── data/                         # Market data loading and feature engineering
│
├── strategies/                   # Alpha signals, CPCV validation, and options pricing
│
├── execution/                    # Order execution and TCA simulation
│
├── risk/                         # Risk management and position sizing
│
├── results/                      # Backtest outputs and performance metrics
│
├── auth/                         # Authentication modules
│
├── logs/                         # Execution logs
│
├── app.py                        # Interactive Flask dashboard application
├── research_dashboard.py         # Advanced research and analytics interface
├── config.py                     # Configuration settings
└── README.md
```

---

### **How It Works**

### **1. Alpha Generation & Preprocessing**

* Loads OHLCV market datasets
* Computes multi-factor indicators (e.g., Amihud Illiquidity)
* Normalizes cross-sectional scores and evaluates predictive IC
* Performs rolling window factor timing

The system applies quantitative adjustments to improve signal robustness:

| Adjustment Technique | Purpose                           |
| -------------------- | --------------------------------- |
| Zero-Volume Masking  | Prevents NaN propagation          |
| Rank Correlation     | Handles non-normal distributions  |
| Factor Decay         | Excludes insignificant alpha      |
| Index Embargoing     | Severs autoregressive correlation |
| Volatility Scaling   | Normalizes position sizing        |

---

### **2. Quantitative Architecture**

Uses advanced statistical modeling and derivatives pricing for automated strategy validation.

```python
from scipy.stats import spearmanr
from arch import arch_model

# Example of volatility modeling
vol_model = arch_model(returns, vol='EGARCH', p=1, o=1, q=1)
res = vol_model.fit()
forecasts = res.forecast(horizon=1)
```

---

### **3. Execution Microstructure (TCA)**

The system integrates Transaction Cost Analysis to produce realistic backtest net returns:
* Models bid-ask bounce and spread crossing
* Deducts explicit nonlinear NSE fee structures and taxes
* Calculates Almgren-Chriss temporary market impact

---

### **Model Performance**

| Metric    | Objective |
| --------- | ------ |
| Deflated Sharpe Ratio  | > 0.95   |
| Expected Maximum Sharpe | Realistic   |
| ICIR    | Significant   |
| Drawdown  | Minimized |

---

### **Interactive Application Deployment**

The project features an interactive **Flask Web Application** designed with clean UI aesthetics, enabling users to visualize charts, run backtests, and explore strategies.

#### **To Launch the Platform Locally:**
```powershell
python app.py
```

---

### **Technology Stack**

| Category             | Tools                     |
| -------------------- | ------------------------- |
| Programming          | Python                    |
| Quantitative Finance | Pandas, SciPy, Arch       |
| Data Processing      | NumPy                     |
| Backtesting          | Custom Event-Driven Engine|
| Web Framework        | Flask                     |
| Frontend             | HTML/JS (TradingView lightweight charts) |

---

### **Getting Started**

### **1. Clone Repository**

```bash
git clone https://github.com/yourusername/quandao.git

cd quandao/quandao_public
```

---

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### **3. Launch Dashboard**

```bash
python app.py
```

Open:

```bash
http://localhost:5000
```

---

### **4. Run Research Backtests**

```bash
python -m strategies.mp
```

---

### **Example Use Case**

An institutional quantitative desk can use this system to:

1. Validate new trading alpha using strictly purged cross-validation
2. Assess market impact and realistic transaction costs before live trading
3. Dynamically size positions inversely to conditional EGARCH volatility forecasts
4. Analyze performance and drawdowns visually through the Flask dashboard

---

### **Future Improvements**

* Real-time FIX protocol execution engine
* Machine learning ensemble factor blending
* Distributed computing for high-frequency tick data processing

---

### **Contributors**

* **Jegadeesh D** — Quantitative architecture design, statistical modeling, volatility forecasting, TCA implementation, and algorithmic strategy development

---

### **License**

MIT License
