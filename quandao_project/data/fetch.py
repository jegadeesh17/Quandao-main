"""
Fetch historical data from Fyers, preprocess and store in TimescaleDB (Postgres).
Usage examples (non-interactive; good for cron):
  # Full backfill (daily + 1m):
  python backend/data/fetch.py --mode backfill --symbol "NSE:NIFTY50-INDEX"

  # Backfill only daily:
  python backend/data/fetch.py --mode backfill_daily --symbol "NSE:NIFTY50-INDEX"

  # Backfill only 1m:
  python backend/data/fetch.py --mode backfill_1m --symbol "NSE:NIFTY50-INDEX"

  #backfill for N days 
  python backend/data/fetch.py --mode backfill_1m --symbol "NSE:NIFTY50-INDEX" --days 3

  # Update today's data (to be run at 16:00 IST daily after market close):
  python backend/data/fetch.py --mode daily_update --symbol "NSE:NIFTY50-INDEX"

  # Catch up from the latest available date in DB till today:
  python backend/data/fetch.py --mode catchup --symbol "NSE:NIFTY50-INDEX"

  # Update all symbols and resolutions present in DB for the last N days:
  python backend/data/fetch.py --mode update_all --days 7
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import pandas as pd
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# --- PROJECT ROOT SETUP ---
_file_path = Path(__file__).resolve()
_project_root = str(_file_path.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)

# Load .env from project root
load_dotenv(os.path.join(_project_root, ".env"))

IST = timezone(timedelta(hours=5, minutes=30))
DAILY_START_DATE = "1990-01-01"
ONE_MIN_START_DATE = "2017-01-01"

DAILY_RESOLUTIONS = {"D", "1D"}
INTRADAY_RESOLUTIONS = {
    "5S", "10S", "15S", "30S", "45S",
    "1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240",
}

# load_dotenv(".env") # Replaced by project root logic above

CLIENT_ID = os.getenv("APP_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL in .env")

LOG_PATH = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_PATH, exist_ok=True)

fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID, is_async=False, token=ACCESS_TOKEN, log_path=LOG_PATH
)

def ensure_table_and_hypertable(conn):
    """Create table if not exists and convert to hypertable"""
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
        try:
            cur.execute(create_extension_sql)
        except Exception:
            pass
        try:
            cur.execute(create_hypertable_sql)
        except Exception:
            pass
    conn.commit()

def upsert_candles(conn, rows):
    """Bulk upsert with deduplication"""
    if not rows:
        return

    seen = set()
    unique_rows = []
    for r in rows:
        key = (r[0], r[1], r[2])  # (time, symbol, resolution)
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
        execute_values(cur, insert_sql, unique_rows, page_size=1000)
    conn.commit()

def candles_to_df(candles, resolution, readable=False):
    """Convert Fyers candles to DataFrame with tz-aware UTC datetime"""
    if readable:
        df = pd.DataFrame(candles)
        return df

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.drop(columns=["timestamp"])
    df = df[["time", "open", "high", "low", "close", "volume"]]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def format_for_upsert(df, symbol, resolution):
    rows = []
    for _, r in df.iterrows():
        t = r["time"]
        if pd.isna(t):
            continue
        rows.append((t.to_pydatetime(), symbol, resolution, float(r["open"]), float(r["high"]),
                     float(r["low"]), float(r["close"]), float(r["volume"])))
    return rows

def fetch_between(symbol, from_date, to_date, resolution, readable=False):
    dt_from = datetime.strptime(from_date, "%Y-%m-%d")
    dt_to = datetime.strptime(to_date, "%Y-%m-%d")
    if dt_to < dt_from:
        raise ValueError("to_date must be >= from_date")

    max_days = 366 if resolution in DAILY_RESOLUTIONS else 100 if resolution in INTRADAY_RESOLUTIONS else None
    if max_days is None:
        raise ValueError("Unsupported resolution")

    all_candles = []
    current_start = dt_from
    while current_start <= dt_to:
        current_end = min(current_start + timedelta(days=max_days-1), dt_to)
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": current_start.strftime("%Y-%m-%d"),
            "range_to": current_end.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        print(f"[FYERS] Fetching {symbol} {resolution} from {data['range_from']} to {data['range_to']}")
        resp = fyers.history(data=data)
        if "candles" in resp:
            all_candles.extend(resp["candles"])
        else:
            print(f"[FYERS] Warning: no candles for {data['range_from']} to {data['range_to']}")
        current_start = current_end + timedelta(days=1)

    df = candles_to_df(all_candles, resolution, readable)
    return df

def resample_from_1m(df_1m, minutes):
    if minutes == 1:
        return df_1m.copy()
    df = df_1m.copy().set_index("time").sort_index()
    agg = df.resample(f"{minutes}min", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna(subset=["open"]).reset_index()
    return agg

def backfill_all(symbol, conn, days=None):
    today_ist = datetime.now(IST).date()

    daily_start = DAILY_START_DATE
    one_min_start = ONE_MIN_START_DATE
    if days is not None:
        daily_start = (today_ist - timedelta(days=days)).strftime("%Y-%m-%d")
        one_min_start = daily_start

    print(f"[BACKFILL] Daily from {daily_start} to {today_ist}")
    df_daily = fetch_between(symbol, daily_start, today_ist.strftime("%Y-%m-%d"), "D")
    if not df_daily.empty:
        upsert_candles(conn, format_for_upsert(df_daily, symbol, "D"))
        print(f"[DB] Inserted/updated {len(df_daily)} daily rows")

    print(f"[BACKFILL] 1m from {one_min_start} to {today_ist}")
    df_1m = fetch_between(symbol, one_min_start, today_ist.strftime("%Y-%m-%d"), "1")
    if not df_1m.empty:
        upsert_candles(conn, format_for_upsert(df_1m, symbol, "1"))
        print(f"[DB] Inserted/updated {len(df_1m)} 1m rows")

def daily_update(symbol, conn):
    today_ist = datetime.now(IST).date()
    from_date = today_ist.strftime("%Y-%m-%d")

    df_daily = fetch_between(symbol, from_date, from_date, "D")
    if not df_daily.empty:
        upsert_candles(conn, format_for_upsert(df_daily, symbol, "D"))
        print(f"[DB] Updated {len(df_daily)} daily rows for {from_date}")

    df_1m = fetch_between(symbol, from_date, from_date, "1")
    if not df_1m.empty:
        upsert_candles(conn, format_for_upsert(df_1m, symbol, "1"))
        print(f"[DB] Updated {len(df_1m)} 1m rows for {from_date}")

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
    today_ist = datetime.now(IST).date()
    today_str = today_ist.strftime("%Y-%m-%d")

    # Catchup Daily
    latest_d = get_latest_timestamp(conn, symbol, "D")
    start_d = latest_d.astimezone(IST).strftime("%Y-%m-%d") if latest_d else DAILY_START_DATE
    print(f"[CATCHUP] Daily from {start_d} to {today_str}")
    df_daily = fetch_between(symbol, start_d, today_str, "D")
    if not df_daily.empty:
        upsert_candles(conn, format_for_upsert(df_daily, symbol, "D"))
        print(f"[DB] Inserted/updated {len(df_daily)} daily rows from {start_d} to {today_str}")
    else:
        print(f"[DB] No new daily data found from {start_d} to {today_str}")

    # Catchup 1m
    latest_1m = get_latest_timestamp(conn, symbol, "1")
    start_1m = latest_1m.astimezone(IST).strftime("%Y-%m-%d") if latest_1m else ONE_MIN_START_DATE
    print(f"[CATCHUP] 1m from {start_1m} to {today_str}")
    df_1m = fetch_between(symbol, start_1m, today_str, "1")
    if not df_1m.empty:
        upsert_candles(conn, format_for_upsert(df_1m, symbol, "1"))
        print(f"[DB] Inserted/updated {len(df_1m)} 1m rows from {start_1m} to {today_str}")
    else:
        print(f"[DB] No new 1m data found from {start_1m} to {today_str}")

def update_all(conn, days):
    today_ist = datetime.now(IST).date()
    start_date = (today_ist - timedelta(days=days)).strftime("%Y-%m-%d")
    today_str = today_ist.strftime("%Y-%m-%d")

    sql = "SELECT DISTINCT symbol, resolution FROM market_candles;"
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    print(f"[UPDATE ALL] Found {len(rows)} total symbol/resolution pairs in DB.")
    for symbol, resolution in rows:
        # Only update Fyers symbols (must contain a colon e.g. NSE:NIFTY50-INDEX)
        if ":" not in symbol:
            continue

        print(f"\n[UPDATE ALL] Updating {symbol} ({resolution}) for last {days} days ({start_date} to {today_str})")
        try:
            df = fetch_between(symbol, start_date, today_str, resolution)
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
    parser.add_argument("--symbol", required=False, help='e.g. "NSE:NIFTY50-INDEX"')
    parser.add_argument("--days", type=int, help="Number of past days to fetch (overrides default start dates)")
    args = parser.parse_args()

    if args.mode != "update_all" and not args.symbol:
        parser.error(f"--symbol is required for mode {args.mode}")

    conn = psycopg2.connect(DATABASE_URL)
    ensure_table_and_hypertable(conn)

    if args.mode == "update_all":
        days = args.days if args.days is not None else 7
        update_all(conn, days)
    elif args.mode == "backfill":
        backfill_all(args.symbol, conn, getattr(args, "days", None))
    elif args.mode == "backfill_daily":
        today_ist = datetime.now(IST).date()
        start_date = (today_ist - timedelta(days=args.days)).strftime("%Y-%m-%d") if getattr(args, "days", None) else DAILY_START_DATE
        df_daily = fetch_between(args.symbol, start_date, today_ist.strftime("%Y-%m-%d"), "D")
        if not df_daily.empty:
            upsert_candles(conn, format_for_upsert(df_daily, args.symbol, "D"))
    elif args.mode == "backfill_1m":
        today_ist = datetime.now(IST).date()
        start_date = (today_ist - timedelta(days=args.days)).strftime("%Y-%m-%d") if getattr(args, "days", None) else ONE_MIN_START_DATE
        df_1m = fetch_between(args.symbol, start_date, today_ist.strftime("%Y-%m-%d"), "1")
        if not df_1m.empty:
            upsert_candles(conn, format_for_upsert(df_1m, args.symbol, "1"))
    elif args.mode == "daily_update":
        daily_update(args.symbol, conn)
    elif args.mode == "catchup":
        catchup(args.symbol, conn)

    conn.close()

if __name__ == "__main__":
    main()
