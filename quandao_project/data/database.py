"""
quandao_project/data/database.py
==================================
The ONLY module in the project that is allowed to issue SQL statements.

Responsibilities
----------------
* Open connections (psycopg2 + SQLAlchemy)
* Create / migrate schema
* Write: upsert_candles, helpers
* Read: all load_ohlcv_* variants (daily, weekly, monthly, yearly,
        intraday N-min, and raw-1m for Forex ORB strategies)

The rest of the codebase calls these functions exclusively.
No SQL should appear anywhere else.
"""

from __future__ import annotations

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text

from quandao_project.config import Settings


# ===========================================================================
# Connections
# ===========================================================================

def get_connection():
    """Return a raw psycopg2 connection (used by write operations)."""
    return psycopg2.connect(Settings.DATABASE_URL)


def get_engine():
    """Return a SQLAlchemy engine (used by read/pd.read_sql operations)."""
    return create_engine(Settings.DATABASE_URL)


# ===========================================================================
# Schema
# ===========================================================================

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_candles (
    time        TIMESTAMPTZ      NOT NULL,
    symbol      TEXT             NOT NULL,
    resolution  TEXT             NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    PRIMARY KEY (symbol, time, resolution)
);
"""

_CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS timescaledb;"
_CREATE_HYPERTABLE_SQL = (
    "SELECT create_hypertable('market_candles', 'time', if_not_exists => TRUE);"
)

_CREATE_FACTOR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_factors (
    time            TIMESTAMPTZ      NOT NULL,
    symbol          TEXT             NOT NULL,
    factor_momentum DOUBLE PRECISION,
    factor_lowvol   DOUBLE PRECISION,
    factor_amihud   DOUBLE PRECISION,
    PRIMARY KEY (symbol, time)
);
"""
_CREATE_FACTOR_HYPERTABLE_SQL = (
    "SELECT create_hypertable('market_factors', 'time', if_not_exists => TRUE);"
)


def ensure_schema(conn) -> None:
    """Create market_candles/market_factors tables + TimescaleDB hypertables if absent."""
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)
        cur.execute(_CREATE_FACTOR_TABLE_SQL)
        for stmt in (_CREATE_EXTENSION_SQL, _CREATE_HYPERTABLE_SQL, _CREATE_FACTOR_HYPERTABLE_SQL):
            try:
                cur.execute(stmt)
            except Exception:
                conn.rollback()
    conn.commit()


# ===========================================================================
# Write helpers
# ===========================================================================

_UPSERT_SQL = """
INSERT INTO market_candles (time, symbol, resolution, open, high, low, close, volume)
VALUES %s
ON CONFLICT (symbol, time, resolution)
DO UPDATE SET
    open   = EXCLUDED.open,
    high   = EXCLUDED.high,
    low    = EXCLUDED.low,
    close  = EXCLUDED.close,
    volume = EXCLUDED.volume;
"""


def upsert_candles(conn, rows: list[tuple], page_size: int = 1000) -> None:
    """
    Bulk-upsert deduplicated OHLCV rows into market_candles.

    Parameters
    ----------
    conn      : psycopg2 connection
    rows      : list of (time, symbol, resolution, open, high, low, close, volume)
    page_size : execute_values batch size (use 5000 for MT5 bulk loads)
    """
    if not rows:
        return

    seen: set = set()
    unique: list[tuple] = []
    for r in rows:
        key = (r[0], r[1], r[2])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    with conn.cursor() as cur:
        execute_values(cur, _UPSERT_SQL, unique, page_size=page_size)
    conn.commit()


def get_latest_timestamp(conn, symbol: str, resolution: str):
    """Return the most recent stored datetime for (symbol, resolution), or None."""
    sql = """
    SELECT max(time) FROM market_candles
    WHERE symbol = %s AND resolution = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, resolution))
        res = cur.fetchone()
        if res and res[0]:
            return res[0]
    return None


def get_distinct_pairs(conn) -> list[tuple[str, str]]:
    """Return all (symbol, resolution) pairs currently in the database."""
    sql = "SELECT DISTINCT symbol, resolution FROM market_candles;"
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()

_UPSERT_FACTORS_SQL = """
INSERT INTO market_factors (time, symbol, factor_momentum, factor_lowvol, factor_amihud)
VALUES %s
ON CONFLICT (symbol, time)
DO UPDATE SET
    factor_momentum = EXCLUDED.factor_momentum,
    factor_lowvol   = EXCLUDED.factor_lowvol,
    factor_amihud   = EXCLUDED.factor_amihud;
"""

def upsert_factors(conn, rows: list[tuple], page_size: int = 1000) -> None:
    """
    Bulk-upsert deduplicated factor rows into market_factors.

    Parameters
    ----------
    conn      : psycopg2 connection
    rows      : list of (time, symbol, factor_momentum, factor_lowvol, factor_amihud)
    """
    if not rows:
        return

    seen: set = set()
    unique: list[tuple] = []
    for r in rows:
        key = (r[0], r[1])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    with conn.cursor() as cur:
        execute_values(cur, _UPSERT_FACTORS_SQL, unique, page_size=page_size)
    conn.commit()


# ===========================================================================
# Read helpers  (previously in backend/data/load.py)
# ===========================================================================
# All functions return a DataFrame sorted ascending by `time`.
# The caller never writes SQL — they call these functions.

def load_ohlcv_day(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Load daily OHLCV bars for *symbol* in [from_date, to_date)."""
    engine = get_engine()
    sql = text("""
        SELECT time, open, high, low, close, volume
        FROM   market_candles
        WHERE  symbol     = :symbol
        AND    resolution = 'D'
        AND    time      >= :from_date
        AND    time      <  :to_date
        ORDER  BY time
    """)
    df = pd.read_sql(sql, engine, params={"symbol": symbol,
                                           "from_date": from_date,
                                           "to_date": to_date})
    return df.sort_values("time").reset_index(drop=True)


def load_ohlcv_week(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Load weekly OHLCV bars for *symbol*."""
    df = load_ohlcv_day(symbol, from_date, to_date)
    if df.empty:
        return df
    df.set_index("time", inplace=True)
    df = df.resample("W-MON").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    return df


def load_ohlcv_month(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Load monthly OHLCV bars for *symbol*."""
    df = load_ohlcv_day(symbol, from_date, to_date)
    if df.empty:
        return df
    df.set_index("time", inplace=True)
    df = df.resample("ME").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    return df


def load_ohlcv_year(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Load yearly OHLCV bars for *symbol*."""
    df = load_ohlcv_day(symbol, from_date, to_date)
    if df.empty:
        return df
    df.set_index("time", inplace=True)
    df = df.resample("YE").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    return df


def load_ohlcv_nmin(
    symbol: str,
    resolution: str,
    from_date: str,
    to_date: str,
    session_start: str = "09:15",
    session_end: str = "15:30",
) -> pd.DataFrame:
    """
    Load N-minute OHLCV bars for *symbol* (Indian intraday).
    """
    engine = get_engine()
    sql = text("""
        SELECT time, open, high, low, close, volume
        FROM   market_candles
        WHERE  symbol     = :symbol
        AND    resolution = '1'
        AND    time      >= :from_date
        AND    time      <  :to_date
        AND    time      >= time::date + :session_start ::time
        AND    time      <= time::date + :session_end   ::time
        ORDER  BY time
    """)
    df = pd.read_sql(sql, engine, params={
        "symbol":       symbol,
        "from_date":    from_date,
        "to_date":      to_date,
        "session_start": session_start,
        "session_end":   session_end,
    })
    
    if df.empty:
        return df
        
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("Asia/Kolkata")
    df.set_index("time", inplace=True)
    
    freq = resolution.replace(" minutes", "min")
    df = df.resample(freq).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    return df


def load_ohlcv(
    symbol: str,
    resolution: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """
    Unified OHLCV loader — dispatches to the correct underlying function.

    Parameters
    ----------
    resolution : ``'D'``, ``'W'``, ``'M'``, ``'Y'``, or an integer string
                 (``'1'``…``'240'``) for intraday minutes.

    Returns
    -------
    pd.DataFrame or None if the resolution is unrecognised.
    """
    if resolution == "D":
        return load_ohlcv_day(symbol, from_date, to_date)
    if resolution == "W":
        return load_ohlcv_week(symbol, from_date, to_date)
    if resolution == "M":
        return load_ohlcv_month(symbol, from_date, to_date)
    if resolution == "Y":
        return load_ohlcv_year(symbol, from_date, to_date)
    try:
        res_int = int(resolution)
        if 1 <= res_int <= 240:
            return load_ohlcv_nmin(symbol, f"{res_int} minutes", from_date, to_date)
    except ValueError:
        pass
    return pd.DataFrame()


def load_ohlcv_forex(
    symbol: str,
    from_date: str,
    to_date: str,
    session_open_hour: int = 0,
    session_close_hour: int = 24,
) -> pd.DataFrame:
    """
    Load raw 1-minute Forex OHLCV from the database.

    Forex data is stored in ``market_candles`` with ``resolution='1'`` by
    the MT5 ingestion pipeline.  Strategies resample internally as needed.

    Parameters
    ----------
    symbol             : e.g. ``"GBPUSD"``
    from_date          : start timestamp with tz, e.g. ``"2020-01-01 00:00:00+00:00"``
    to_date            : end   timestamp with tz
    session_open_hour  : UTC hour filter – 0 means no filter
    session_close_hour : UTC hour filter – 24 means no filter

    Returns
    -------
    pd.DataFrame – columns: time (UTC datetime), open, high, low, close, volume.
    Empty DataFrame if no data found.
    """
    engine = get_engine()
    sql = text("""
        SELECT time, open, high, low, close, volume
        FROM   market_candles
        WHERE  symbol     = :symbol
        AND    resolution = '1'
        AND    time      >= :from_date
        AND    time      <  :to_date
        ORDER  BY time
    """)
    df = pd.read_sql(sql, engine, params={"symbol": symbol,
                                           "from_date": from_date,
                                           "to_date": to_date})
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    if session_open_hour > 0 or session_close_hour < 24:
        hour = df["time"].dt.hour
        df = df[
            (hour >= session_open_hour) & (hour < session_close_hour)
        ].reset_index(drop=True)

    return df


def load_factors(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """Load factors for *symbol* in [from_date, to_date)."""
    engine = get_engine()
    sql = text("""
        SELECT time, factor_momentum, factor_lowvol, factor_amihud
        FROM   market_factors
        WHERE  symbol = :symbol
        AND    time  >= :from_date
        AND    time  <  :to_date
        ORDER  BY time
    """)
    df = pd.read_sql(sql, engine, params={"symbol": symbol,
                                           "from_date": from_date,
                                           "to_date": to_date})
    return df.sort_values("time").reset_index(drop=True)
