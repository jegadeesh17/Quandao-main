# Quandao Quantitative Trading & Research Platform

## Project Overview
Institutional-grade quantitative research, backtesting, and automated execution platform. The system operates on dual layers of alpha generation: a Moon-Phase Opening Range Breakout (ORB) intraday execution engine for NIFTY50 index trading, and a sector-neutralized multi-factor cross-sectional stock screening model. It provides advanced validation frameworks including Combinatorial Purged Cross-Validation (CPCV) and the Deflated Sharpe Ratio (DSR) to mathematically control for multiple testing and selection bias.

The system ingests historical and real-time tick/candle data from the Fyers API and MetaTrader 5, stores it in a structured PostgreSQL database utilizing TimescaleDB hypertables, runs quantitative pricing and liquidity factor models, and exposes interactive analytics through a Streamlit dashboard.

## Key Features
* **Moon-Phase Opening Range Breakout (ORB) Strategy**: Combines directional astronomical moon cycles (Long bias after New Moon, Short bias after Full Moon) with 10-minute intraday opening breakout channels, VWAP confirmation filters, and Average True Range (ATR) dynamic stop losses.
* **Combinatorial Purged Cross-Validation (CPCV)**: Divides historical return series into combinations of train/test splits to construct independent out-of-sample performance paths, mitigating traditional cross-validation leakage and bias.
* **Deflated Sharpe Ratio (DSR)**: Penalizes backtest Sharpe ratios by calculating the statistical significance of strategy performance relative to the number of factor formulations tested.
* **Sector-Neutralized Multi-Factor Stock Screener**: Scores NSE equities on multiple indicators (Amihud Liquidity, SMA-100 Pullback, Quarterly Low Volatility, Volume-Price Trend), neutralizing factor outputs within GICS sectors to remove industry concentration distortion.
* **Programmatic 3-Stage Gatekeeper Filters**: Minimizes manual chart-checking to less than 10% by applying hard fundamental filters (industry-relative P/E, leverage ratios) and technical triggers (trend direction, RSI stabilization channels).
* **Database & Ingestion Pipeline**: Backfills and catchup-syncs daily and 1-minute OHLCV candles, using explicit index mappings to preserve volume fields for accurate liquidity calculations.
* **Execution & State Recovery**: Supports dry-run paper trading and live execution order mapping to the Fyers API with robust daily state recovery files.

## Dataset
* **Source**: Fyers REST API (Indian Equities & Indices), MetaTrader 5 Terminal (Forex & Gold), and local PostgreSQL databases.
* **Coverage**: NIFTY50 index, Nifty 50 constituent equities, Forex majors, and spot Gold.
* **Data Type**: Multi-resolution historical OHLCV data (1-minute, intraday N-min, daily, weekly, monthly) and computed factor matrices.
* **Included Tables**:
  * `market_candles`: Primary time-series table storing date, symbol, resolution, open, high, low, close, and volume.
  * `market_factors`: Relational table holding pre-computed factor metrics per symbol and timestamp.

## Project Structure
```
quandao_project/
│
├── data/                         # Data ingestion and database interfaces
│   ├── database.py               # PostgreSQL connection, schema definitions, and read/write queries
│   ├── data_fetcher.py           # Fyers & MetaTrader 5 API adapters and backfill/catchup routines
│   └── factor_engine.py          # Mathematical engine computing price, volume, and liquidity factors
│
├── strategies/                   # Core alpha generation, filtering, and cross-validation
│   ├── alpha_signals.py          # Stateless pure functions for mathematical factor transformations
│   ├── multi_factor.py           # Composite score calculations and sector neutralization logic
│   ├── cpcv_validation.py        # CPCV splitting and DSR/Probabilistic DSR analytics
│   ├── moonphase_orb.py          # Dual-layer Moon-Phase ORB intraday trading logic
│   └── base_strategy.py          # Strategy base class defining common operations
│
├── risk/                         # Risk management
│   └── risk_manager.py           # Position sizing, capital allocations, and safety rules
│
├── execution/                    # Broker order execution
│   └── order_executor.py         # Order routing, broker payloads, and transaction logging
│
├── research_dashboard.py         # Streamlit Quantitative Research Dashboard and Screener
├── main.py                       # CLI tool for backtests and system triggers
└── requirements.txt              # Standard python dependency requirements
```

## How It Works

### 1. Data Ingestion & Relational Setup
The system establishes connections to the local PostgreSQL database using SQLAlchemy and Psycopg2. Historical daily and 1-minute data are fetched from broker APIs and bulk-upserted into TimescaleDB hypertables:
```python
# From quandao_project/data/database.py
from sqlalchemy import create_engine

# Get SQLAlchemy connection engine
def get_engine():
    return create_engine(Settings.DATABASE_URL)

# Unified candle loading wrapper
def load_ohlcv(symbol: str, resolution: str, from_date: str, to_date: str):
    engine = get_engine()
    # Pulls OHLCV data from market_candles hypertable
    ...
```

### 2. Quantitative Analytics & Factor Neutralization
Active alpha factors are evaluated as stateless pure functions. In the cross-sectional ranking engine, factor scoring is neutralized by GICS sectors to remove industry concentration bias before executing the Multi-Factor composite score:
```python
# From quandao_project/strategies/multi_factor.py
def neutralize_factor(df, factor_col, group_col='Sector'):
    """De-means factor values within GICS sector groups."""
    group_means = df.groupby(group_col)[factor_col].transform('mean')
    return df[factor_col] - group_means
```

### 3. Interactive Streamlit Dashboard Deployment
The research dashboard serves a multi-tab UI for exploring cross-validation results and screening Nifty 50 swing trade setups:
```bash
streamlit run quandao_project/research_dashboard.py
```

## Technology Stack
| Category | Tools |
| --- | --- |
| Programming | Python |
| Database Engine | PostgreSQL (with TimescaleDB extension) |
| Database Connection | SQLAlchemy, Psycopg2-binary |
| Broker APIs | fyers-apiv3, MetaTrader5 |
| Data Processing | Pandas, NumPy, SciPy |
| Web Dashboard Framework | Streamlit, Flask |
| Visualization | Matplotlib, Seaborn |

## Getting Started

### 1. Setup Database
Ensure PostgreSQL is running locally. Create a new database named `quandao`:
```sql
CREATE DATABASE quandao;
```

### 2. Install Dependencies
Install all package requirements in your virtual environment:
```bash
pip install -r quandao_project/requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root directory and add your database and broker credentials:
```env
DB_USER=postgres
DB_PASSWORD=your_postgres_password
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=quandao

FYERS_APP_ID=your_fyers_app_id
FYERS_SECRET_KEY=your_fyers_secret_key
FYERS_ACCESS_TOKEN=your_current_daily_access_token
```

### 4. Ingest Data
Authenticate your broker session and execute the backfill scripts to populate the database:
```bash
# Authorize Fyers session via browser login
python -m quandao_project.data.data_fetcher --source fyers --mode login

# Backfill historical daily and minute candle data for Nifty 50 constituents
python -m quandao_project.data.data_fetcher --source fyers --mode nifty50_backfill
```

### 5. Launch the Dashboard & Live Bot
Run the Streamlit research dashboard or trigger paper/live trading scripts:
```bash
# Start Streamlit GUI
streamlit run quandao_project/research_dashboard.py

# Run the Moon-Phase ORB strategy in dry-run (paper) mode
python -m quandao_project.strategies.moonphase_orb --dry-run
```

## Example Use Case
Quantitative developers and portfolio managers can utilize the platform to:
1. **Analyze Strategy Selection Bias**: Run CPCV validation across historical data to evaluate whether out-of-sample Sharpe ratios hold up against deflated thresholds.
2. **Execute Stock Screening**: Identify Nifty 50 companies with strong sector-relative Momentum, Amihud Liquidity, and Low Volatility while filtering out overvalued or over-leveraged names.
3. **Execute Intraday Options Trading**: Run the automated bot to place opening breakout orders with strict risk guards (max 1 trade per day and ATR stop loss).

## Future Improvements
* **Advanced Machine Learning Inputs**: Integrating LSTM models to predict breakout success probability.
* **Multi-Broker Integrations**: Expanding execution adaptors to Interactive Brokers and Zerodha.
* **Options Chain Ingestion**: Ingesting real-time option chain Greeks to execute synthetic option structures on breakouts.

## Contributors
* **Jegadeesh D** — Aspiring Quant and Software Engineer.
## License
MIT License
