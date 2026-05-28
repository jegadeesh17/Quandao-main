"""
quandao_project/data/data_fetcher.py
======================================
The ONLY module in the project that is allowed to call external broker APIs.

Responsibilities
----------------
* Fetch raw OHLCV from Fyers REST API (Indian equities / indices)
* Fetch raw OHLCV from MetaTrader 5 local terminal (Forex)
* Provide pipeline modes: backfill, catchup, daily_update, update_all
* Batch backfill for symbol universes (e.g. Nifty 50)

All credentials, start-dates, and rate limits come from Settings.
No SQL lives here — writes are delegated entirely to database.py.

Usage examples (run from project root)
---------------------------------------
  python -m quandao_project.data.data_fetcher --source fyers --mode backfill    --symbol "NSE:NIFTY50-INDEX"
  python -m quandao_project.data.data_fetcher --source fyers --mode backfill_daily --symbol "NSE:NIFTY50-INDEX"
  python -m quandao_project.data.data_fetcher --source fyers --mode catchup     --symbol "NSE:NIFTY50-INDEX"
  python -m quandao_project.data.data_fetcher --source fyers --mode daily_update --symbol "NSE:NIFTY50-INDEX"
  python -m quandao_project.data.data_fetcher --source fyers --mode update_all  --days 7
  python -m quandao_project.data.data_fetcher --source fyers --mode nifty50_backfill

  python -m quandao_project.data.data_fetcher --source forex --mode backfill    --symbol GBPUSD
  python -m quandao_project.data.data_fetcher --source forex --mode catchup     --symbol GBPUSD
  python -m quandao_project.data.data_fetcher --source forex --mode update_all  --days 7
"""

from __future__ import annotations

import argparse
import time
import pandas as pd
from datetime import datetime, timedelta, timezone

from quandao_project.config import Settings
from quandao_project.data.database import (
    get_connection,
    ensure_schema,
    upsert_candles,
    get_latest_timestamp,
    get_distinct_pairs,
)

# ---------------------------------------------------------------------------
# Timezone singletons
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

# ---------------------------------------------------------------------------
# Fyers resolution constants
# ---------------------------------------------------------------------------
DAILY_RESOLUTIONS = {"D", "1D"}
INTRADAY_RESOLUTIONS = {
    "5S", "10S", "15S", "30S", "45S",
    "1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240",
}

# ---------------------------------------------------------------------------
# MT5 resolution map
# ---------------------------------------------------------------------------
def _mt5_timeframes():
    import MetaTrader5 as mt5
    return {"1": mt5.TIMEFRAME_M1, "D": mt5.TIMEFRAME_D1}

# ---------------------------------------------------------------------------
# Nifty 50 constituent universe (used by nifty50_backfill mode)
# ---------------------------------------------------------------------------
NIFTY_50_SYMBOLS: list[str] = [
    "NSE:ADANIENT-EQ",  "NSE:ADANIPORTS-EQ", "NSE:APOLLOHOSP-EQ", "NSE:ASIANPAINT-EQ",
    "NSE:AXISBANK-EQ",  "NSE:BAJAJ-AUTO-EQ", "NSE:BAJFINANCE-EQ", "NSE:BAJAJFINSV-EQ",
    "NSE:BEL-EQ",       "NSE:BPCL-EQ",       "NSE:BHARTIARTL-EQ", "NSE:BRITANNIA-EQ",
    "NSE:CIPLA-EQ",     "NSE:COALINDIA-EQ",  "NSE:DRREDDY-EQ",    "NSE:EICHERMOT-EQ",
    "NSE:GRASIM-EQ",    "NSE:HCLTECH-EQ",    "NSE:HDFCBANK-EQ",   "NSE:HDFCLIFE-EQ",
    "NSE:HEROMOTOCO-EQ","NSE:HINDALCO-EQ",   "NSE:HINDUNILVR-EQ", "NSE:ICICIBANK-EQ",
    "NSE:ITC-EQ",       "NSE:INDUSINDBK-EQ", "NSE:INFY-EQ",       "NSE:JSWSTEEL-EQ",
    "NSE:KOTAKBANK-EQ", "NSE:LT-EQ",         "NSE:M&M-EQ",        "NSE:MARUTI-EQ",
    "NSE:NTPC-EQ",      "NSE:NESTLEIND-EQ",  "NSE:ONGC-EQ",       "NSE:POWERGRID-EQ",
    "NSE:RELIANCE-EQ",  "NSE:SBILIFE-EQ",    "NSE:SHRIRAMFIN-EQ", "NSE:SBIN-EQ",
    "NSE:SUNPHARMA-EQ", "NSE:TCS-EQ",        "NSE:TATACONSUM-EQ", "NSE:TATAMOTORS-EQ",
    "NSE:TATASTEEL-EQ", "NSE:TECHM-EQ",      "NSE:TITAN-EQ",      "NSE:ULTRACEMCO-EQ",
    "NSE:WIPRO-EQ",     "NSE:TRENT-EQ",
]


# ===========================================================================
# Internal helpers
# ===========================================================================

def _format_for_upsert(df: pd.DataFrame, symbol: str, resolution: str) -> list[tuple]:
    """Convert a cleaned OHLCV DataFrame to upsert-ready tuples."""
    rows = []
    for _, r in df.iterrows():
        t = r["time"]
        if pd.isna(t):
            continue
        rows.append((
            t.to_pydatetime(), symbol, resolution,
            float(r["open"]), float(r["high"]),
            float(r["low"]),  float(r["close"]), float(r["volume"]),
        ))
    return rows


def _today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def _today_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


# ===========================================================================
# Fyers  — REST API polling
# ===========================================================================

def save_env_var(key: str, value: str) -> None:
    """Update or add a key=value pair in the .env file."""
    env_path = Settings.ENV_PATH
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    new_lines, updated = [], False

    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines))


def login_fyers() -> None:
    """Run full Fyers login flow via browser authentication."""
    from fyers_apiv3 import fyersModel
    import webbrowser
    
    print("[FYERS AUTH] Starting browser authorization flow...")
    client_id = Settings.FYERS_APP_ID
    secret_key = Settings.FYERS_APP_SECRET
    redirect_uri = Settings.FYERS_REDIRECT_URI

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    auth_url = session.generate_authcode()
    print(f"\n[FYERS AUTH] Open this URL in your browser to authorize:\n{auth_url}")
    webbrowser.open(auth_url)

    auth_code = input("\n[FYERS AUTH] Paste the 'auth_code' from the redirected URL: ").strip()
    session.set_token(auth_code)
    token_response = session.generate_token()

    if token_response.get("access_token"):
        save_env_var("ACCESS_TOKEN", token_response["access_token"])
        print("[FYERS AUTH] ACCESS_TOKEN updated.")
    if token_response.get("refresh_token"):
        save_env_var("REFRESH_TOKEN", token_response["refresh_token"])
        print("[FYERS AUTH] REFRESH_TOKEN updated.")

    Settings.reload()


def _build_fyers_client():
    """Construct an authenticated Fyers REST client from Settings."""
    from fyers_apiv3 import fyersModel
    Settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    return fyersModel.FyersModel(
        client_id=Settings.FYERS_APP_ID,
        is_async=False,
        token=Settings.FYERS_ACCESS_TOKEN,
        log_path=str(Settings.LOG_DIR),
    )



def _candles_to_df(candles: list) -> pd.DataFrame:
    """Convert raw Fyers candle list -> cleaned UTC DataFrame."""
    if not candles:
        return pd.DataFrame()
    # Explicitly extract standard format fields from index values
    parsed = []
    for c in candles:
        if isinstance(c, list) and len(c) >= 6:
            parsed.append({
                "timestamp": c[0],
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5] # Index 5 explicitly
            })
        elif isinstance(c, dict):
            parsed.append({
                "timestamp": c.get("timestamp"),
                "open": c.get("open"),
                "high": c.get("high"),
                "low": c.get("low"),
                "close": c.get("close"),
                "volume": c.get("volume")
            })
    if parsed:
        df = pd.DataFrame(parsed)
    else:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.drop(columns=["timestamp"], errors="ignore")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["time", "open", "high", "low", "close", "volume"]]


def fetch_fyers_ltp(symbol: str, fyers_client=None) -> float:
    """Fetch the last traded price (LTP) for a symbol via Fyers REST API."""
    client = fyers_client or _build_fyers_client()
    resp = client.quotes(data={"symbols": symbol})
    if resp.get("s") == "ok" and resp.get("d"):
        val = resp["d"][0].get("v", {})
        return float(val.get("lp", 0.0))
    else:
        raise RuntimeError(f"Failed to fetch quote: {resp.get('message')}")


def fetch_fyers(
    symbol: str,
    from_date: str,
    to_date: str,
    resolution: str = "D",
) -> pd.DataFrame:
    """
    Poll the Fyers REST history endpoint for *symbol* across date range.

    Automatically splits the request into chunks that respect Fyers API
    limits (366 days for daily; 100 days for intraday).

    Parameters
    ----------
    symbol     : Fyers symbol, e.g. ``"NSE:NIFTY50-INDEX"``
    from_date  : ``"YYYY-MM-DD"``
    to_date    : ``"YYYY-MM-DD"``
    resolution : ``"D"`` | ``"1"`` | ``"5"`` | … (see INTRADAY_RESOLUTIONS)

    Returns
    -------
    pd.DataFrame – columns: time (UTC tz-aware), open, high, low, close, volume
    """
    if resolution in DAILY_RESOLUTIONS:
        max_days = 366
    elif resolution in INTRADAY_RESOLUTIONS:
        max_days = 100
    else:
        raise ValueError(f"Unsupported Fyers resolution: {resolution!r}")

    fyers = _build_fyers_client()
    dt_from = datetime.strptime(from_date, "%Y-%m-%d")
    dt_to   = datetime.strptime(to_date,   "%Y-%m-%d")

    all_candles: list = []
    current = dt_from
    while current <= dt_to:
        end = min(current + timedelta(days=max_days - 1), dt_to)
        payload = {
            "symbol":      symbol,
            "resolution":  resolution,
            "date_format": "1",
            "range_from":  current.strftime("%Y-%m-%d"),
            "range_to":    end.strftime("%Y-%m-%d"),
            "cont_flag":   "1",
        }
        print(f"[FYERS] {symbol} {resolution}: {payload['range_from']} -> {payload['range_to']}")
        resp = fyers.history(data=payload)
        if "candles" in resp:
            all_candles.extend(resp["candles"])
        else:
            print(f"[FYERS] Warning – no candles {payload['range_from']} to {payload['range_to']}")
        current = end + timedelta(days=1)

    return _candles_to_df(all_candles)


# ===========================================================================
# MetaTrader 5  — local terminal polling
# ===========================================================================

def fetch_forex(
    symbol: str,
    from_date: str,
    to_date: str,
    resolution: str = "1",
) -> pd.DataFrame:
    """
    Pull OHLCV data from a running MetaTrader 5 terminal.

    Data is fetched in day-chunks to avoid MT5 memory constraints.

    Parameters
    ----------
    symbol     : MT5 symbol, e.g. ``"GBPUSD"``
    from_date  : ``"YYYY-MM-DD"`` (UTC)
    to_date    : ``"YYYY-MM-DD"`` (UTC)
    resolution : ``"1"`` (1-minute) | ``"D"`` (daily)

    Returns
    -------
    pd.DataFrame – columns: time (UTC tz-aware), open, high, low, close, volume
    """
    import MetaTrader5 as mt5
    tf_map = _mt5_timeframes()

    if resolution not in tf_map:
        raise ValueError(f"Unsupported MT5 resolution: {resolution!r}")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    if not mt5.symbol_select(symbol, True):
        raise ValueError(f"MT5 symbol not available: {symbol!r}")

    timeframe  = tf_map[resolution]
    dt_from    = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=UTC)
    dt_to      = datetime.strptime(to_date,   "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    chunk_days = 100 if resolution == "1" else 2000

    all_dfs: list[pd.DataFrame] = []
    current = dt_from
    while current < dt_to:
        end   = min(current + timedelta(days=chunk_days), dt_to)
        rates = mt5.copy_rates_range(symbol, timeframe, current, end)
        if rates is not None and len(rates) > 0:
            all_dfs.append(pd.DataFrame(rates))
        current = end

    if not all_dfs:
        print(f"[MT5] Warning – no candles for {symbol} {from_date} to {to_date}")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df.drop_duplicates(subset=["time"], inplace=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["time", "open", "high", "low", "close", "volume"]]


# ===========================================================================
# Fyers pipeline modes
# ===========================================================================

def fyers_backfill(symbol: str, conn, days: int | None = None) -> None:
    """Full backfill: daily + 1-minute from configured start dates (or last N days)."""
    today = _today_ist()
    daily_start = Settings.DAILY_START_DATE
    onemin_start = Settings.ONE_MIN_START_DATE

    if days is not None:
        cutoff = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
        daily_start = cutoff
        onemin_start = cutoff

    print(f"[FYERS BACKFILL] Daily   {daily_start} -> {today}")
    df = fetch_fyers(symbol, daily_start, today, "D")
    if not df.empty:
        upsert_candles(conn, _format_for_upsert(df, symbol, "D"))
        print(f"[DB] {len(df)} daily rows -> {symbol}")

    print(f"[FYERS BACKFILL] 1-min   {onemin_start} -> {today}")
    df = fetch_fyers(symbol, onemin_start, today, "1")
    if not df.empty:
        upsert_candles(conn, _format_for_upsert(df, symbol, "1"))
        print(f"[DB] {len(df)} 1m rows -> {symbol}")


def fyers_backfill_daily(symbol: str, conn, days: int | None = None) -> None:
    """Backfill only daily bars."""
    today = _today_ist()
    start = Settings.DAILY_START_DATE
    if days is not None:
        start = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetch_fyers(symbol, start, today, "D")
    if not df.empty:
        upsert_candles(conn, _format_for_upsert(df, symbol, "D"))
        print(f"[DB] {len(df)} daily rows -> {symbol}")


def fyers_backfill_1m(symbol: str, conn, days: int | None = None) -> None:
    """Backfill only 1-minute bars."""
    today = _today_ist()
    start = Settings.ONE_MIN_START_DATE
    if days is not None:
        start = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetch_fyers(symbol, start, today, "1")
    if not df.empty:
        upsert_candles(conn, _format_for_upsert(df, symbol, "1"))
        print(f"[DB] {len(df)} 1m rows -> {symbol}")


def fyers_daily_update(symbol: str, conn) -> None:
    """Refresh today's daily + 1-minute bars (run at 16:00 IST after close)."""
    today = _today_ist()
    for res in ("D", "1"):
        df = fetch_fyers(symbol, today, today, res)
        if not df.empty:
            upsert_candles(conn, _format_for_upsert(df, symbol, res))
            print(f"[DB] Updated {len(df)} {res} rows for {today} -> {symbol}")


def fyers_catchup(symbol: str, conn) -> None:
    """Fetch from the last stored timestamp to today (fills any gap)."""
    today = _today_ist()
    for res, default_start in (("D", Settings.DAILY_START_DATE),
                                ("1", Settings.ONE_MIN_START_DATE)):
        latest = get_latest_timestamp(conn, symbol, res)
        start = (
            latest.astimezone(IST).strftime("%Y-%m-%d")
            if latest else default_start
        )
        print(f"[FYERS CATCHUP] {res}: {start} -> {today}")
        df = fetch_fyers(symbol, start, today, res)
        if not df.empty:
            upsert_candles(conn, _format_for_upsert(df, symbol, res))
            print(f"[DB] {len(df)} {res} rows -> {symbol}")
        else:
            print(f"[DB] No new {res} data for {symbol}")


def fyers_update_all(conn, days: int = 7) -> None:
    """Re-fetch the last *days* for every Fyers symbol/resolution already in the DB."""
    today = _today_ist()
    start = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
    pairs = [p for p in get_distinct_pairs(conn) if ":" in p[0]]   # Fyers symbols contain ":"
    print(f"[FYERS UPDATE ALL] {len(pairs)} pair(s) | {start} -> {today}")
    for symbol, resolution in pairs:
        if resolution not in (DAILY_RESOLUTIONS | INTRADAY_RESOLUTIONS):
            continue
        print(f"  {symbol} ({resolution})")
        try:
            df = fetch_fyers(symbol, start, today, resolution)
            if not df.empty:
                upsert_candles(conn, _format_for_upsert(df, symbol, resolution))
                print(f"  [DB] {len(df)} rows")
        except Exception as exc:
            print(f"  [ERROR] {exc}")


def fyers_nifty50_backfill(conn) -> None:
    """
    Daily backfill for all 50 Nifty 50 constituents.

    Uses a 1-second inter-request sleep to respect Fyers rate limits.
    """
    today = _today_ist()
    total = len(NIFTY_50_SYMBOLS)
    print(f"\n{'='*55}\n  Nifty 50 Backfill — {total} symbols\n{'='*55}\n")

    for i, symbol in enumerate(NIFTY_50_SYMBOLS, 1):
        print(f"[{i:02d}/{total}] {symbol}")
        try:
            df = fetch_fyers(symbol, Settings.DAILY_START_DATE, today, "D")
            if not df.empty:
                upsert_candles(conn, _format_for_upsert(df, symbol, "D"))
                print(f"       -> {len(df)} rows stored")
            else:
                print(f"       -> [WARN] no data")
        except Exception as exc:
            print(f"       -> [ERROR] {exc}")
        time.sleep(1)

    print(f"\n{'='*55}\n  Nifty 50 Backfill COMPLETE\n{'='*55}\n")


# ===========================================================================
# Forex / MT5 pipeline modes
# ===========================================================================

def forex_backfill(symbol: str, conn) -> None:
    """Full backfill for a Forex pair: daily + 1-minute."""
    today = _today_utc()
    for res, start in (("D", Settings.FOREX_DAILY_START_DATE),
                        ("1", Settings.FOREX_ONE_MIN_START_DATE)):
        print(f"[FOREX BACKFILL] {res}: {start} -> {today}")
        df = fetch_forex(symbol, start, today, res)
        if not df.empty:
            ps = 5000 if res == "1" else 1000
            upsert_candles(conn, _format_for_upsert(df, symbol, res), page_size=ps)
            print(f"[DB] {len(df)} {res} rows -> {symbol}")


def forex_daily_update(symbol: str, conn) -> None:
    """Refresh the last 2 days of daily + 1-minute Forex data."""
    today = _today_utc()
    from_date = (datetime.now(UTC) - timedelta(days=2)).strftime("%Y-%m-%d")
    import MetaTrader5 as mt5
    try:
        for res in ("D", "1"):
            df = fetch_forex(symbol, from_date, today, res)
            if not df.empty:
                ps = 5000 if res == "1" else 1000
                upsert_candles(conn, _format_for_upsert(df, symbol, res), page_size=ps)
                print(f"[DB] Updated {len(df)} {res} rows -> {symbol}")
    finally:
        mt5.shutdown()


def forex_catchup(symbol: str, conn) -> None:
    """Fetch Forex data from last stored timestamp to today."""
    import MetaTrader5 as mt5
    today = _today_utc()
    try:
        for res, default_start in (("D", Settings.FOREX_DAILY_START_DATE),
                                    ("1", Settings.FOREX_ONE_MIN_START_DATE)):
            latest = get_latest_timestamp(conn, symbol, res)
            start = (
                latest.astimezone(UTC).strftime("%Y-%m-%d")
                if latest else default_start
            )
            print(f"[FOREX CATCHUP] {res}: {start} -> {today}")
            df = fetch_forex(symbol, start, today, res)
            if not df.empty:
                ps = 5000 if res == "1" else 1000
                upsert_candles(conn, _format_for_upsert(df, symbol, res), page_size=ps)
                print(f"[DB] {len(df)} rows -> {symbol}")
            else:
                print(f"[DB] No new {res} data for {symbol}")
    finally:
        mt5.shutdown()


def forex_update_all(conn, days: int = 7) -> None:
    """Re-fetch the last *days* for every Forex symbol/resolution already in the DB."""
    import MetaTrader5 as mt5
    tf_map = _mt5_timeframes()
    today = _today_utc()
    start = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
    pairs = [(s, r) for s, r in get_distinct_pairs(conn)
             if ":" not in s and r in tf_map]          # Forex symbols have no ":"
    print(f"[FOREX UPDATE ALL] {len(pairs)} pair(s) | {start} -> {today}")
    try:
        for symbol, resolution in pairs:
            print(f"  {symbol} ({resolution})")
            try:
                df = fetch_forex(symbol, start, today, resolution)
                if not df.empty:
                    ps = 5000 if resolution == "1" else 1000
                    upsert_candles(conn, _format_for_upsert(df, symbol, resolution),
                                   page_size=ps)
                    print(f"  [DB] {len(df)} rows")
            except Exception as exc:
                print(f"  [ERROR] {exc}")
    finally:
        mt5.shutdown()


# ===========================================================================
# Public one-call pipeline helpers  (used by main.py and scripts)
# ===========================================================================

def ingest_fyers(symbol: str, from_date: str, to_date: str,
                 resolution: str = "D") -> int:
    """Fetch from Fyers and upsert. Returns row count."""
    df = fetch_fyers(symbol, from_date, to_date, resolution)
    if df.empty:
        return 0
    conn = get_connection()
    ensure_schema(conn)
    rows = _format_for_upsert(df, symbol, resolution)
    upsert_candles(conn, rows)
    conn.close()
    print(f"[DB] {len(rows)} rows upserted for {symbol} ({resolution})")
    return len(rows)


def ingest_forex(symbol: str, from_date: str, to_date: str,
                 resolution: str = "1") -> int:
    """Fetch from MT5 and upsert. Returns row count."""
    import MetaTrader5 as mt5
    try:
        df = fetch_forex(symbol, from_date, to_date, resolution)
        if df.empty:
            return 0
        conn = get_connection()
        ensure_schema(conn)
        rows = _format_for_upsert(df, symbol, resolution)
        upsert_candles(conn, rows, page_size=5000)
        conn.close()
        print(f"[DB] {len(rows)} rows upserted for {symbol} ({resolution})")
        return len(rows)
    finally:
        mt5.shutdown()


# ===========================================================================
# CLI  (python -m quandao_project.data.data_fetcher …)
# ===========================================================================

def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="data_fetcher",
        description="Quandao data ingestion pipeline",
    )
    parser.add_argument(
        "--source", required=True, choices=["fyers", "forex"],
        help="Data source adapter to use",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=[
            "backfill", "backfill_daily", "backfill_1m",
            "daily_update", "catchup", "update_all", "nifty50_backfill",
            "login",
        ],
    )
    parser.add_argument("--symbol", help='e.g. "NSE:NIFTY50-INDEX" or "GBPUSD"')
    parser.add_argument("--days",   type=int, help="Override default lookback days")
    args = parser.parse_args()

    if args.mode not in ("update_all", "nifty50_backfill", "login") and not args.symbol:
        parser.error(f"--symbol is required for mode '{args.mode}'")

    if args.source == "fyers" and args.mode == "login":
        login_fyers()
        return

    conn = get_connection()
    ensure_schema(conn)

    try:
        if args.source == "fyers":
            if args.mode == "backfill":
                fyers_backfill(args.symbol, conn, args.days)
            elif args.mode == "backfill_daily":
                fyers_backfill_daily(args.symbol, conn, args.days)
            elif args.mode == "backfill_1m":
                fyers_backfill_1m(args.symbol, conn, args.days)
            elif args.mode == "daily_update":
                fyers_daily_update(args.symbol, conn)
            elif args.mode == "catchup":
                fyers_catchup(args.symbol, conn)
            elif args.mode == "update_all":
                fyers_update_all(conn, days=args.days or 7)
            elif args.mode == "nifty50_backfill":
                fyers_nifty50_backfill(conn)

        elif args.source == "forex":
            if args.mode == "backfill":
                forex_backfill(args.symbol, conn)
            elif args.mode == "daily_update":
                forex_daily_update(args.symbol, conn)
            elif args.mode == "catchup":
                forex_catchup(args.symbol, conn)
            elif args.mode == "update_all":
                forex_update_all(conn, days=args.days or 7)
            else:
                parser.error(f"Mode '{args.mode}' not supported for --source forex")
    finally:
        conn.close()


if __name__ == "__main__":
    _cli()
