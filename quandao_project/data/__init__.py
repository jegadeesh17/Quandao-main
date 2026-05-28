"""
quandao_project/data/__init__.py
----------------------------------
Public surface of the data layer.

The rest of the codebase (strategies, risk, execution, main) should only
ever import from this package — never from database.py or data_fetcher.py
directly.  This gives us a single place to rename or swap implementations.

Read API  (load from DB)
------------------------
    from quandao_project.data import load_ohlcv, load_ohlcv_forex

Write / Ingest API  (fetch from broker → write to DB)
------------------------------------------------------
    from quandao_project.data import ingest_fyers, ingest_forex

Low-level raw fetch (no DB touch, returns DataFrame)
-----------------------------------------------------
    from quandao_project.data import fetch_fyers, fetch_forex

Connection helpers
------------------
    from quandao_project.data import get_connection, get_engine
"""

# -- Read (SQL) -------------------------------------------------------------
from quandao_project.data.database import (
    get_connection,
    get_engine,
    ensure_schema,
    upsert_candles,
    get_latest_timestamp,
    get_distinct_pairs,
    # Typed loaders
    load_ohlcv,
    load_ohlcv_day,
    load_ohlcv_week,
    load_ohlcv_month,
    load_ohlcv_year,
    load_ohlcv_nmin,
    load_ohlcv_forex,
)

# -- Fetch (broker API) -----------------------------------------------------
from quandao_project.data.data_fetcher import (
    fetch_fyers,
    fetch_fyers_ltp,
    fetch_forex,
    ingest_fyers,
    ingest_forex,
    # Fyers pipeline modes
    fyers_backfill,
    fyers_backfill_daily,
    fyers_backfill_1m,
    fyers_daily_update,
    fyers_catchup,
    fyers_update_all,
    fyers_nifty50_backfill,
    # Forex / MT5 pipeline modes
    forex_backfill,
    forex_daily_update,
    forex_catchup,
    forex_update_all,
    # Constants
    NIFTY_50_SYMBOLS,
)

__all__ = [
    # connections
    "get_connection", "get_engine",
    # schema / write
    "ensure_schema", "upsert_candles",
    "get_latest_timestamp", "get_distinct_pairs",
    # read
    "load_ohlcv", "load_ohlcv_day", "load_ohlcv_week",
    "load_ohlcv_month", "load_ohlcv_year", "load_ohlcv_nmin",
    "load_ohlcv_forex",
    # raw fetch
    "fetch_fyers", "fetch_fyers_ltp", "fetch_forex",
    # pipeline ingest
    "ingest_fyers", "ingest_forex",
    # Fyers modes
    "fyers_backfill", "fyers_backfill_daily", "fyers_backfill_1m",
    "fyers_daily_update", "fyers_catchup", "fyers_update_all",
    "fyers_nifty50_backfill",
    # Forex modes
    "forex_backfill", "forex_daily_update", "forex_catchup", "forex_update_all",
    # constants
    "NIFTY_50_SYMBOLS",
]
