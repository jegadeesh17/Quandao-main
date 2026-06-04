"""
Fetch historical Forex data from MetaTrader 5, preprocess and store in TimescaleDB (Postgres).
Usage examples (Terminal must be running):
  # Full backfill (daily + 1m):
  python fetch_forex.py --mode backfill --symbol "EURUSD"

  # Backfill only daily:
  python fetch_forex.py --mode backfill_daily --symbol "GBPUSD"

  # Backfill only 1m:
  python fetch_forex.py --mode backfill_1m --symbol "USDJPY"

  # Update today's data (run end of day):
  python fetch_forex.py --mode daily_update --symbol "EURUSD"

  # Catch up from the latest available date in DB till today:
  python fetch_forex.py --mode catchup --symbol "EURUSD"

  # Update all forex symbols and resolutions present in DB for the last N days:
  python backend/data/fetch_forex.py --mode update_all --days 7
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# Forex data goes much further back natively in MT5 than Indian equities.
DAILY_START_DATE = "2000-01-01" 
ONE_MIN_START_DATE = "2015-01-01"

# Map string resolutions to MT5 internal timeframe constants
MT5_TIMEFRAMES = {
    "1": mt5.TIMEFRAME_M1,
    "D": mt5.TIMEFRAME_D1
}

# --- PROJECT ROOT SETUP ---
_file_path = Path(__file__).resolve()
_project_root = str(_file_path.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)

# Load .env from project root
load_dotenv(os.path.join(_project_root, ".env"))
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL in .env")

# --- DATABASE SETUP (Unchanged) ---
def ensure_table_and_hypertable(conn):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS market_candles (
        time TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        resolution TEXT NOT NULL,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume DOUBLE PRECISION,
        PRIMARY KEY (symbol, time, resolution)
    );
    """
    create_extension_sql = "CREATE EXTENSION IF NOT EXISTS timescaledb;"
    create_hypertable_sql = "SELECT create_hypertable('market_candles', 'time', if_not_exists => TRUE);"

    with conn.cursor() as cur:
        cur.execute(create_table_sql)
        try: cur.execute(create_extension_sql)
        except Exception: pass
        try: cur.execute(create_hypertable_sql)
        except Exception: pass
    conn.commit()

def upsert_candles(conn, rows):
    if not rows: return
    seen = set()
    unique_rows = []
    for r in rows:
        key = (r[0], r[1], r[2])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    insert_sql = """
    INSERT INTO market_candles (time, symbol, resolution, open, high, low, close, volume)
    VALUES %s
    ON CONFLICT (symbol, time, resolution)
    DO UPDATE SET
      open = EXCLUDED.open,
      high = EXCLUDED.high,
      low = EXCLUDED.low,
      close = EXCLUDED.close,
      volume = EXCLUDED.volume;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, unique_rows, page_size=5000) # Increased page size for MT5 speed
    conn.commit()

def format_for_upsert(df, symbol, resolution):
    rows = []
    for _, r in df.iterrows():
        t = r["time"]
        if pd.isna(t): continue
        rows.append((t.to_pydatetime(), symbol, resolution, float(r["open"]), float(r["high"]),
                     float(r["low"]), float(r["close"]), float(r["volume"])))
    return rows

# --- MT5 FETCHING LOGIC ---
def fetch_between_mt5(symbol, from_date_str, to_date_str, resolution):
    """Fetches data from MT5 local terminal using DataFrame chunking to preserve column names."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed, error code: {mt5.last_error()}")

    if not mt5.symbol_select(symbol, True):
        raise ValueError(f"Symbol {symbol} not found in MT5 or couldn't be selected.")

    timeframe = MT5_TIMEFRAMES[resolution]
    
    dt_from = datetime.strptime(from_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt_to = datetime.strptime(to_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    print(f"[MT5] Fetching {symbol} {resolution} from {from_date_str} to {to_date_str}")
    
    chunk_days = 100 if resolution == "1" else 2000 
    all_dfs = [] # Store DataFrames instead of raw tuples
    
    current_start = dt_from
    while current_start < dt_to:
        current_end = min(current_start + timedelta(days=chunk_days), dt_to)
        
        rates = mt5.copy_rates_range(symbol, timeframe, current_start, current_end)
        
        if rates is not None and len(rates) > 0:
            # Convert to DataFrame immediately to preserve MT5 column names
            chunk_df = pd.DataFrame(rates)
            all_dfs.append(chunk_df)
            
        current_start = current_end

    if not all_dfs:
        print(f"[MT5] Warning: no candles found for {from_date_str} to {to_date_str}")
        return pd.DataFrame()

    # Safely stitch all the chunk DataFrames together
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Now 'time' definitely exists
    df.drop_duplicates(subset=['time'], inplace=True)
    
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    return df

# --- BACKFILL ROUTINES ---
def backfill_all(symbol, conn):
    today_utc = datetime.now(timezone.utc).date()

    print(f"[BACKFILL] Daily from {DAILY_START_DATE} to {today_utc}")
    df_daily = fetch_between_mt5(symbol, DAILY_START_DATE, today_utc.strftime("%Y-%m-%d"), "D")
    if not df_daily.empty:
        upsert_candles(conn, format_for_upsert(df_daily, symbol, "D"))
        print(f"[DB] Inserted/updated {len(df_daily)} daily rows")

    print(f"[BACKFILL] 1m from {ONE_MIN_START_DATE} to {today_utc}")
    # Note: Pulling 10 years of 1M data might take a few seconds and use some RAM (~3.5M rows)
    df_1m = fetch_between_mt5(symbol, ONE_MIN_START_DATE, today_utc.strftime("%Y-%m-%d"), "1")
    if not df_1m.empty:
        upsert_candles(conn, format_for_upsert(df_1m, symbol, "1"))
        print(f"[DB] Inserted/updated {len(df_1m)} 1m rows")

def daily_update(symbol, conn):
    today_utc = datetime.now(timezone.utc).date()
    # Go back 2 days just to ensure we catch late-arriving broker data and overlaps safely
    from_date = (today_utc - timedelta(days=2)).strftime("%Y-%m-%d")
    to_date = today_utc.strftime("%Y-%m-%d")

    df_daily = fetch_between_mt5(symbol, from_date, to_date, "D")
    if not df_daily.empty:
        upsert_candles(conn, format_for_upsert(df_daily, symbol, "D"))
        print(f"[DB] Updated {len(df_daily)} daily rows from {from_date}")

    df_1m = fetch_between_mt5(symbol, from_date, to_date, "1")
    if not df_1m.empty:
        upsert_candles(conn, format_for_upsert(df_1m, symbol, "1"))
        print(f"[DB] Updated {len(df_1m)} 1m rows from {from_date}")

def get_latest_timestamp(conn, symbol, resolution):
    """Query the database for the latest timestamp for a given symbol and resolution."""
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

def catchup(symbol, conn):
    today_utc = datetime.now(timezone.utc).date()
    today_str = today_utc.strftime("%Y-%m-%d")

    # Catchup Daily
    latest_d = get_latest_timestamp(conn, symbol, "D")
    start_d = latest_d.astimezone(timezone.utc).strftime("%Y-%m-%d") if latest_d else DAILY_START_DATE
    print(f"[CATCHUP] Daily from {start_d} to {today_str}")
    df_daily = fetch_between_mt5(symbol, start_d, today_str, "D")
    if not df_daily.empty:
        upsert_candles(conn, format_for_upsert(df_daily, symbol, "D"))
        print(f"[DB] Inserted/updated {len(df_daily)} daily rows from {start_d} to {today_str}")
    else:
        print(f"[DB] No new daily data found from {start_d} to {today_str}")

    # Catchup 1m
    latest_1m = get_latest_timestamp(conn, symbol, "1")
    start_1m = latest_1m.astimezone(timezone.utc).strftime("%Y-%m-%d") if latest_1m else ONE_MIN_START_DATE
    print(f"[CATCHUP] 1m from {start_1m} to {today_str}")
    df_1m = fetch_between_mt5(symbol, start_1m, today_str, "1")
    if not df_1m.empty:
        upsert_candles(conn, format_for_upsert(df_1m, symbol, "1"))
        print(f"[DB] Inserted/updated {len(df_1m)} 1m rows from {start_1m} to {today_str}")
    else:
        print(f"[DB] No new 1m data found from {start_1m} to {today_str}")

def update_all(conn, days):
    today_utc = datetime.now(timezone.utc).date()
    start_date = (today_utc - timedelta(days=days)).strftime("%Y-%m-%d")
    today_str = today_utc.strftime("%Y-%m-%d")

    sql = "SELECT DISTINCT symbol, resolution FROM market_candles;"
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    print(f"[UPDATE ALL] Found {len(rows)} total symbol/resolution pairs in DB.")
    for symbol, resolution in rows:
        # Only update MT5 Forex symbols (must NOT contain a colon e.g. EURUSD)
        if ":" in symbol:
            continue

        if resolution not in MT5_TIMEFRAMES:
            continue

        print(f"\n[UPDATE ALL] Updating {symbol} ({resolution}) for last {days} days ({start_date} to {today_str})")
        try:
            df = fetch_between_mt5(symbol, start_date, today_str, resolution)
            if not df.empty:
                upsert_candles(conn, format_for_upsert(df, symbol, resolution))
                print(f"[DB] Updated {len(df)} rows for {symbol} ({resolution})")
            else:
                print(f"[DB] No data found for {symbol} ({resolution}) in given range.")
        except Exception as e:
            print(f"[ERROR] Failed to update {symbol} ({resolution}): {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backfill", "backfill_daily", "backfill_1m", "daily_update", "catchup", "update_all"], required=True)
    parser.add_argument("--symbol", required=False, help='e.g. "EURUSD"')
    parser.add_argument("--days", type=int, help="Number of past days to fetch")
    args = parser.parse_args()

    if args.mode != "update_all" and not args.symbol:
        parser.error(f"--symbol is required for mode {args.mode}")

    conn = psycopg2.connect(DATABASE_URL)
    ensure_table_and_hypertable(conn)

    try:
        if args.mode == "update_all":
            days = args.days if args.days is not None else 7
            update_all(conn, days)
        elif args.mode == "backfill":
            backfill_all(args.symbol, conn)
        elif args.mode == "backfill_daily":
            today_utc = datetime.now(timezone.utc).date()
            df_daily = fetch_between_mt5(args.symbol, DAILY_START_DATE, today_utc.strftime("%Y-%m-%d"), "D")
            if not df_daily.empty:
                upsert_candles(conn, format_for_upsert(df_daily, args.symbol, "D"))
                print(f"[DB] Inserted/updated {len(df_daily)} daily rows")
        elif args.mode == "backfill_1m":
            today_utc = datetime.now(timezone.utc).date()
            df_1m = fetch_between_mt5(args.symbol, ONE_MIN_START_DATE, today_utc.strftime("%Y-%m-%d"), "1")
            if not df_1m.empty:
                upsert_candles(conn, format_for_upsert(df_1m, args.symbol, "1"))
                print(f"[DB] Inserted/updated {len(df_1m)} 1m rows")
        elif args.mode == "daily_update":
            daily_update(args.symbol, conn)
        elif args.mode == "catchup":
            catchup(args.symbol, conn)
    finally:
        # Always gracefully shut down MT5 connection and close DB
        mt5.shutdown()
        conn.close()

if __name__ == "__main__":
    main()