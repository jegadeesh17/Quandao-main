# QUANDAO TRADING SYSTEM: COMPLETE TECHNICAL GUIDE & ARCHITECTURE

## 1. System Overview & Core Philosophy

The Quandao Trading System is a modular, institutional-grade quantitative trading framework engineered to eliminate the complexities and overlapping dependencies of traditional algorithmic codebases. Built on a strict clean-architecture design philosophy, it completely divorces data ingestion, signal generation, risk management, and broker execution into distinct, non-overlapping pipelines. This structural rigidity ensures that research elements and experimental modeling can transition into live production environments seamlessly, without the risk of contaminating the execution logic or creating circular dependencies.

Central to the strategy layer is the paradigm of "Stateless Pure Functions." Every technical indicator, Opening Range Breakout (ORB) module, and GARCH volatility filter is designed as a pure mathematical transformation that takes a pandas DataFrame as input and yields a mutated DataFrame or scalar output without triggering any external side effects. By enforcing a rule where the strategy layer cannot interact with databases, perform file I/O, or execute broker API calls, the system intrinsically prevents look-ahead bias and eliminates memory leaks typically caused by caching or persistent state in long-running loops. The state of the strategy is entirely deterministic based solely on the data it is fed at runtime.

The linear flow of data enforces this safety model sequentially:
**Data Ingestion -> Signal Generation -> Risk Assessment -> Broker Execution -> Central Orchestration.**
First, the database provides a pristine slice of historical market data. Second, the strategy layer processes this data through pure functions to generate a strictly defined +1, -1, or 0 signal. Third, the risk manager evaluates this signal against equity limits, ATR stops, and fixed fractional sizing mechanics. Fourth, if approved, the execution layer translates the resulting payload into broker-specific network requests. Finally, a central orchestrator conducts these sequential steps in an isolated, lightweight loop.

## 2. Directory Tree & Codebase Map

```text
quandao_project/
├── config/
│   └── settings.py
├── data/
│   ├── database.py
│   └── data_fetcher.py
├── execution/
│   └── order_executor.py
├── logs/
├── risk/
│   └── risk_manager.py
├── strategies/
│   ├── alpha_signals.py
│   ├── base_strategy.py
│   ├── nifty_orb.py
│   ├── tt_entries.py
│   └── weekly_orb_forex.py
├── main.py
└── requirements.txt
```

- `config/settings.py`: The single source of truth for global state, system paths, and environment variable loading. No other file is permitted to use `os.getenv`.
- `data/database.py`: The exclusive repository for SQL statements. It handles TimescaleDB schemas, execution of batch upsert optimizations, and all data reading utilities.
- `data/data_fetcher.py`: The external API aggregator. It handles polling data from Fyers REST API and local MT5 terminals to ingest raw OHLCV information into the database layer.
- `strategies/alpha_signals.py`: The stateless pure mathematical library. It houses indicator calculations, ORB boundaries, and signal logic without relying on any external I/O.
- `risk/risk_manager.py`: The un-bypasable safety block. It calculates fixed fractional sizing mechanics, handles ATR multiplier math, and validates maximum loss allowances before allowing any payload to reach execution.
- `execution/order_executor.py`: The terminal network endpoint. It receives purely mathematical approved payloads and adapts them into the exact API parameters required by either the Fyers broker, the MT5 engine, or a local JSON simulator.
- `main.py`: The lightweight linear execution loop conductor. It imports elements from all other modules and runs them in a strict top-to-bottom sequence.

## 3. Core Component Deep-Dive & Functionality

### Database & Fetcher

**Database Operations (`data/database.py`):**
- `upsert_candles(conn, rows, page_size=1000)`: Executes an efficient TimescaleDB `ON CONFLICT DO UPDATE` bulk insert for OHLCV rows. The `rows` parameter expects a list of deduplicated tuples containing time, symbol, resolution, open, high, low, close, and volume.
- `load_ohlcv(symbol, resolution, from_date, to_date)`: The unified data loading method. It dynamically routes to underlying SQL functions (like `load_ohlcv_nmin` or `load_ohlcv_day`) based on the requested string `resolution` parameter ('D', 'W', 'M', 'Y', or integer minutes like '5') and returns an ascendingly sorted pandas DataFrame.

**Fetcher Operations (`data/data_fetcher.py`):**
- `fetch_fyers(symbol, from_date, to_date, resolution)`: Authenticates via the Fyers REST API to pull chunks of raw history data. It internally respects Fyers constraints, splitting requests into 100-day or 366-day loops depending on intraday vs daily resolution.
- `fetch_forex(symbol, from_date, to_date, resolution)`: Connects to a local MetaTrader 5 terminal via COM interoperability. It polls for Forex data, looping over specified UTC dates, and gracefully converts MT5 tick structures into standardized pandas DataFrames.

**CLI Controls (`data_fetcher.py`):**
Data fetching can be driven directly via terminal commands utilizing flags:
- `--source`: Specify `fyers` or `forex`.
- `--mode`: Specify operation type. `backfill` loads maximum history, `catchup` spans from the last database timestamp to today, `daily_update` grabs the latest session, and `update_all` loops through every symbol currently stored.
- `--symbol`: Target instrument like `"NSE:NIFTY50-INDEX"` or `"GBPUSD"`.

### Strategies (`strategies/alpha_signals.py`)

- `atr(df, period=14, ewm=False)`: Calculates the Average True Range by computing the max of current high/low difference, gap from previous close to current high, and gap from previous close to current low, applying a rolling window mean.
- `daily_atr(df, period=14)`: Takes an intraday timeframe DataFrame, dynamically resamples it to daily candles using standard market open/close definitions, computes the ATR, and maps it back onto every minute-tick of the intraday frame to avoid mismatched series lengths.
- `session_vwap(df, freq="D")`: A volume-weighted average price that deterministically resets its cumulative sums exactly at the boundaries defined by `freq` (Daily or Weekly).
- `estimate_garch_volatility(df, vol_model="EGARCH", p=1, o=1, q=1, dist="t")`: Performs rigorous conditional volatility modeling and calculates a 95% Value at Risk (VaR) threshold using the `arch` and `scipy` libraries. It handles the mathematical fitting internally without writing to disk.
- `generate_nifty_orb_signals(df, orb_minutes=15, atr_period=14, max_range_atr_ratio=0.8)`: Constructs the entire logic for a Nifty Opening Range Breakout. It calculates indicators, determines the 15-minute start constraints, checks exhaustion parameters, filters by RVOL/VWAP, and enforces the signal contract.
- `generate_forex_orb_signals(df, orb_minutes=240, atr_period=14, max_range_atr_ratio=0.8)`: A weekly scale equivalent for MT5 forex execution, utilizing 4-hour boundaries anchored to the Sunday/Monday session start.

**The Signal Contract:** 
Functions generating signals mandate a strict integer enforcement on the `"signal"` column: `1` for Long, `-1` for Short, and `0` for Hold/Flat. No probabilities, strings, or complex dictionaries are emitted, ensuring that the risk layer can blindly process a simple directional multiplier.

### Risk (`risk/risk_manager.py`)

- `calculate_position_size(equity, risk_percentage, atr_value, stop_multiplier)`: Computes the precise fractional position size based on current available capital. It determines stop distance by multiplying `atr_value` against `stop_multiplier` and returns the exact fractional unit size required to ensure the total risk strictly equals `equity * risk_percentage`.
- `evaluate_trade_allowance(current_open_positions, max_allowed_positions, daily_loss, max_daily_loss_limit)`: Acts as an un-bypasable hard stop. If current open trades hit the ceiling or if the dynamic daily drawdown exceeds defined thresholds, it instantly returns `False`, neutralizing the strategy.
- `generate_risk_adjusted_order(signal, price, atr_value, equity, settings)`: Central execution packager. If the mathematical parameters pass requirements, it formats a payload containing specific entry, stop-loss, and take-profit float values based on ATR multiples. Crucially, it attaches `"status": "APPROVED"` to the dictionary, acting as a cryptographic seal for the execution layer.

### Execution (`execution/order_executor.py`)

- `PaperJournal`: A local JSON storage simulation engine. When live execution is turned off, this class writes the approved order payload into a persistent `paper_trades.json` dictionary format while stamping it with a UUID and a `"status": "FILLED_PAPER"` marker.
- `FyersAdapter`: A translation class that strips Quandao's `"APPROVED"` payload structure into the exact `{ "symbol", "qty", "side", "productType", "stopLoss" }` dictionary needed to execute an intraday market order over the Fyers v3 REST architecture.
- `MT5Adapter`: Connects directly to the MetaTrader 5 engine, calculating integer definitions for `TRADE_ACTION_DEAL`, `ORDER_TYPE_BUY/SELL`, and deviations. It also contains critical logic to ensure local system time is within 600 seconds of the broker server time before routing the payload.
- `execute_order(order_payload, broker, symbol, paper_trade, fyers_client)`: The central execution router. It validates the presence of `"status": "APPROVED"` on the payload, then routes the logic path either to `PaperJournal`, `FyersAdapter`, or `MT5Adapter` depending on boolean flags and strings.

**Approved Payload Structure:**
```json
{
    "signal": 1,
    "target_qty": 500.25,
    "entry_price": 1.25055,
    "stop_loss": 1.24855,
    "take_profit": 1.25655,
    "status": "APPROVED"
}
```

## 4. End-to-End Operational Workflow

The `main.py` pipeline runs as a strictly defined top-to-bottom sequence when executed:

1. **Initialization:** The script loads the user's defined broker choice and symbol.
2. **Data Catchup:** It routes a terminal command to `data_fetcher.py` running in `--mode catchup`. This synchronizes the local database with missing bars directly from the broker API up to the current minute.
3. **Database Load:** It reaches into TimescaleDB and pulls exactly the last 30 days of data at the required resolution into a pandas DataFrame.
4. **Signal Generation:** The fetched DataFrame is passed directly into a pure function within `alpha_signals.py` (e.g., `generate_nifty_orb_signals`). The function evaluates indicators and returns the frame with a defined signal of 1, -1, or 0.
5. **Risk Assessment:** The latest price, ATR value, and integer signal are extracted and sent to `generate_risk_adjusted_order()`. The risk module calculates fractional sizes. If sizing falls below 0, or if limits are hit, it rejects the payload. If passed, it generates the structured `"APPROVED"` package.
6. **Execution Routing:** The `"APPROVED"` package is delivered to `execute_order()`. Depending on the `paper_trade` parameter, it either commits the trade to `paper_trades.json` or fires a live network request through the respective broker adapter.

**Paper Trading vs Live Execution:**
By default, running `python main.py` operates in safe simulation mode. No live market requests are created; instead, trades are fully mapped, UUID stamped, and stored locally in `paper_trades.json`. 

To switch the framework into live execution mode, append the `--live` flag to bypass the JSON simulator and invoke the live broker API.

**Terminal Run Strings:**
- *Live Fyers Index Trading:*
  `python main.py --symbol "NSE:NIFTY50-INDEX" --broker FYERS --live`
- *Paper MetaTrader 5 Forex Trading:*
  `python main.py --symbol "GBPUSD" --broker MT5`

## 5. Extensibility Playbook (How to Add Future Code)

To expand the Quandao Trading System safely without violating clean-architecture rules, follow these specific checklists:

### How to add a new strategy
1. Open `strategies/alpha_signals.py`.
2. Create a new function titled `generate_[name]_signals(df: pd.DataFrame, ...) -> pd.DataFrame`.
3. Within the function body, utilize the pre-existing indicator functions (like `atr()`, `session_vwap()`) to calculate strategy components on a `.copy()` of the input DataFrame.
4. Construct the mathematical logic to define buy and sell condition masks.
5. Apply these masks to a newly created `"signal"` column, forcing a strictly typed integer array of only `1`, `-1`, or `0`.
6. Return the mutated DataFrame. Do not insert any I/O commands, `print()` statements, database queries, or settings imports within the function.
7. Open `main.py` and import the new function, tying it into Step 3 of the orchestration pipeline.

### How to add a new broker connection
1. Open `execution/order_executor.py`.
2. Create a new adapter class named `[BrokerName]Adapter`.
3. Define a `place_order(self, order_payload: dict, symbol: str) -> dict` method within the class.
4. Construct the broker-specific API package by mapping the standardized values in `order_payload` (e.g., `target_qty`, `stop_loss`) to the exact JSON structure or object type required by the external broker API.
5. Execute the network request using the broker's official SDK or `requests`.
6. Catch the response and map it to either `{"status": "FILLED_LIVE", "ticket_id": id}` or `{"status": "REJECTED_LIVE", "error": msg}`.
7. Update the `execute_order()` function router to detect the new broker string and instantiate the new adapter class dynamically.

### How to add a custom risk rule
1. Open `risk/risk_manager.py`.
2. Navigate to the `evaluate_trade_allowance()` function.
3. Add a new parameter for the custom risk input (e.g., `current_margin_utilization: float`, or `macro_news_flag: bool`).
4. Directly below the existing daily loss checks, insert an explicit `if` statement logic block evaluating the new parameter against a threshold.
5. If the logic trips the risk condition, strictly return `False`.
6. Update the pipeline in `main.py` to correctly query and supply the new parameter into the `evaluate_trade_allowance()` function call prior to attempting order generation.
