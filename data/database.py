"""
quandao_public/data/database.py
==================================
Unified database access layer for Quandao.

WHY THIS MODULE EXISTS:
    The research dashboard imports from quandao_public.data.database.
    This module provides a clean, single-function interface over the existing
    multi-resolution load.py logic — one function to rule them all.

DESIGN:
    - get_connection()  → raw psycopg2 connection (for direct SQL)
    - get_engine()      → SQLAlchemy engine (for pandas read_sql)
    - load_ohlcv()      → unified candle loader (wraps load.py routing)
    - load_fundamentals() → PE, D/E, sector from market_fundamentals table
                            Returns empty DataFrame gracefully if table missing.

USAGE:
    from quandao_public.data.database import load_ohlcv, get_connection

    df = load_ohlcv("NSE:NIFTY50-INDEX", resolution="D",
                    from_date="2023-01-01", to_date="2024-01-01")
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ── Project Root & .env Setup ───────────────────────────────────────────────
# Walk up from this file: database.py → data/ → quandao_public/ → Quandao-main/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# Support both DATABASE_URL (full URL) and individual components
_DATABASE_URL = os.getenv("DATABASE_URL") or (
    "postgresql://{user}:{password}@{host}:{port}/{name}".format(
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=os.getenv("DB_PORT", "5432"),
        name=os.getenv("DB_NAME", "quandao"),
    )
)


# ── Connection Factories ────────────────────────────────────────────────────

def get_connection() -> psycopg2.extensions.connection:
    """
    Return a raw psycopg2 connection to the Quandao PostgreSQL database.

    Use for: direct SQL queries, cursor-based inserts, DDL operations.
    Remember to call conn.close() when done.

    Returns
    -------
    psycopg2.connection
    """
    return psycopg2.connect(_DATABASE_URL)


def get_engine():
    """
    Return a SQLAlchemy engine for pandas read_sql() operations.

    Use for: pd.read_sql(query, engine) — the idiomatic pandas DB pattern.

    Returns
    -------
    sqlalchemy.Engine
    """
    return create_engine(_DATABASE_URL)


# ── Unified OHLCV Loader ────────────────────────────────────────────────────

def load_ohlcv(symbol: str,
               resolution: str,
               from_date: str,
               to_date: str) -> pd.DataFrame:
    """
    Unified OHLCV data loader — single function for all resolutions.

    Delegates to the resolution-specific functions in quandao_public/data/load.py.
    The dashboard and all strategy modules call THIS function — not load.py directly.

    Supported Resolutions:
        'D'  → Daily candles
        'W'  → Weekly (aggregated from daily via TimescaleDB time_bucket)
        'M'  → Monthly
        'Y'  → Yearly
        '1'–'240' → Intraday N-minute candles

    Parameters
    ----------
    symbol     : str - Database symbol, e.g. "NSE:NIFTY50-INDEX", "NSE:RELIANCE-EQ"
    resolution : str - Time resolution (see above)
    from_date  : str - Start date/datetime. Accepts:
                       "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS+05:30"
    to_date    : str - End date/datetime (exclusive upper bound in most queries)

    Returns
    -------
    pd.DataFrame with columns: [time, open, high, low, close, volume]
    Returns empty DataFrame if no data or connection fails.
    """
    # Normalize date strings: if only date given, append IST timestamp
    from_date = _normalize_date_str(from_date, is_end=False)
    to_date   = _normalize_date_str(to_date,   is_end=True)

    try:
        # Import here to avoid circular imports at module load time
        from quandao_public.data.load import load_ohlcv as _load_ohlcv
        df = _load_ohlcv(symbol, resolution, from_date, to_date)
        return df if df is not None else pd.DataFrame()

    except Exception as e:
        print(f"[database.load_ohlcv] Failed for {symbol} ({resolution}): {e}")
        return pd.DataFrame()


def _normalize_date_str(date_str: str, is_end: bool = False) -> str:
    """
    Normalize a date string to 'YYYY-MM-DD HH:MM:SS+05:30' format.

    Accepts: 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS+05:30'
    """
    date_str = date_str.strip()

    # Already has timezone info
    if '+' in date_str or 'Z' in date_str:
        return date_str

    # Has time but no timezone
    if ' ' in date_str and ':' in date_str:
        return date_str + '+05:30'

    # Date only — append start or end of day
    if is_end:
        return date_str + ' 23:59:59+05:30'
    else:
        return date_str + ' 00:00:00+05:30'


# ── Fundamentals Loader ─────────────────────────────────────────────────────

def load_fundamentals(conn, symbols: list) -> pd.DataFrame:
    """
    Load fundamental data (PE ratio, D/E ratio, sector) for a list of symbols.

    Data source: 'market_fundamentals' table in PostgreSQL.
    If the table doesn't exist or query fails, returns an empty DataFrame.
    The dashboard handles the empty case gracefully via hardcoded fallbacks.

    Table Schema (expected):
        symbol TEXT, trailing_pe FLOAT, debt_to_equity FLOAT,
        sector TEXT, updated_at TIMESTAMPTZ

    Parameters
    ----------
    conn    : psycopg2.connection - Active database connection
    symbols : list of str         - List of NSE symbols

    Returns
    -------
    pd.DataFrame with columns: [symbol, trailing_pe, debt_to_equity, sector]
    Empty DataFrame if table missing or query error.
    """
    if not symbols:
        return pd.DataFrame()

    try:
        # Check if table exists
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'market_fundamentals'
                );
            """)
            exists = cur.fetchone()[0]

        if not exists:
            return pd.DataFrame()

        # Load fundamentals for requested symbols
        placeholders = ','.join(['%s'] * len(symbols))
        query = f"""
            SELECT symbol, trailing_pe, debt_to_equity, sector
            FROM   market_fundamentals
            WHERE  symbol IN ({placeholders})
        """
        df = pd.read_sql(query, conn, params=symbols)
        return df

    except Exception as e:
        print(f"[database.load_fundamentals] Query failed: {e}")
        return pd.DataFrame()


# ── Utility: Latest Timestamp ────────────────────────────────────────────────

def get_latest_timestamp(symbol: str, resolution: str):
    """
    Get the most recent candle timestamp in the database for a symbol/resolution.

    Used by the data pipeline to determine where to start a catchup fetch.

    Returns
    -------
    datetime or None
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT max(time) FROM market_candles
                WHERE symbol = %s AND resolution = %s;
            """, (symbol, resolution))
            result = cur.fetchone()
        conn.close()
        return result[0] if result and result[0] else None
    except Exception as e:
        print(f"[database.get_latest_timestamp] Error: {e}")
        return None

