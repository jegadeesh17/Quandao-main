import logging
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# Load .env from the project root (two levels up from quandao_public/data/)
_this_dir     = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_this_dir))
load_dotenv(os.path.join(_project_root, ".env"))

logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL")

# Module-level engine: one connection pool shared across all calls.
# pool_size=5 handles concurrent dashboard requests without exhausting PG connections.
# pool_pre_ping=True detects and discards stale connections before use.
_engine = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
) if DB_URL else None


def load_ohlcv_day(symbol: str,
                   from_date: str,
                   to_date: str) -> pd.DataFrame:
    sql = text("""
    SELECT

    time,
    open,
    high,
    low,
    close,
    volume

    FROM market_candles
    WHERE symbol = :symbol
    AND resolution = 'D'
    AND time >= :from_date
    AND time < :to_date
    """)

    try:
        df = pd.read_sql(
            sql,
            _engine,
            params={
                "symbol":    symbol,
                "from_date": from_date,
                "to_date":   to_date,
            },
        )
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        logger.error("load_ohlcv_day failed for %s: %s", symbol, e)
        return pd.DataFrame()


def load_ohlcv_week(symbol: str,
                    from_date: str,
                    to_date: str) -> pd.DataFrame:
    sql = text("""
    SELECT

    time_bucket(
    '1 week',
    time
    ) AS bucket,

    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume

    FROM market_candles
    WHERE symbol = :symbol
    AND resolution = 'D'
    AND time >= :from_date
    AND time < :to_date

    GROUP BY bucket
    ORDER BY bucket;
    """)

    try:
        df = pd.read_sql(
            sql,
            _engine,
            params={
                "symbol":    symbol,
                "from_date": from_date,
                "to_date":   to_date,
            },
        )
        df = df.rename(columns={"bucket": "time"})
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        logger.error("load_ohlcv_week failed for %s: %s", symbol, e)
        return pd.DataFrame()



def load_ohlcv_month(symbol: str,
                     from_date: str,
                     to_date: str) -> pd.DataFrame:
    sql = text("""
    SELECT

    time_bucket(
    '1 month',
    time
    ) AS bucket,

    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume

    FROM market_candles
    WHERE symbol = :symbol
    AND resolution = 'D'
    AND time >= :from_date
    AND time < :to_date

    GROUP BY bucket
    ORDER BY bucket;
    """)

    try:
        df = pd.read_sql(
            sql,
            _engine,
            params={
                "symbol":    symbol,
                "from_date": from_date,
                "to_date":   to_date,
            },
        )
        df = df.rename(columns={"bucket": "time"})
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        logger.error("load_ohlcv_month failed for %s: %s", symbol, e)
        return pd.DataFrame()


def load_ohlcv_year(symbol: str,
                    from_date: str,
                    to_date: str) -> pd.DataFrame:
    sql = text("""
    SELECT

    time_bucket(
    '1 year',
    time
    ) AS bucket,

    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume

    FROM market_candles
    WHERE symbol = :symbol
    AND resolution = 'D'
    AND time >= :from_date
    AND time < :to_date

    GROUP BY bucket
    ORDER BY bucket;
    """)

    try:
        df = pd.read_sql(
            sql,
            _engine,
            params={
                "symbol":    symbol,
                "from_date": from_date,
                "to_date":   to_date,
            },
        )
        df = df.rename(columns={"bucket": "time"})
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        logger.error("load_ohlcv_year failed for %s: %s", symbol, e)
        return pd.DataFrame()


def load_ohlcv_nmin(symbol: str,
                    resolution: str,
                    from_date: str,
                    to_date: str,
                    session_start: str = '09:15',
                    session_end: str = '15:30') -> pd.DataFrame:
    sql = text("""
    SELECT

    time_bucket(
    :bucket_size,
    time,
    date(time) + time :session_start) AS bucket,

    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume

    FROM market_candles
    WHERE symbol = :symbol
    AND resolution = '1'
    AND time >= :from_date
    AND time < :to_date
    AND time >= date(time) + time :session_start
    AND time <= date(time) + time :session_end

    GROUP BY bucket
    ORDER BY bucket
    """)

    try:
        df = pd.read_sql(
            sql,
            _engine,
            params={
                "symbol":       symbol,
                "bucket_size":  resolution,
                "from_date":    from_date,
                "to_date":      to_date,
                "session_start": session_start,
                "session_end":  session_end,
            },
        )
        df["bucket"] = pd.to_datetime(df["bucket"])
        df["bucket"] = df["bucket"].dt.tz_convert("Asia/Kolkata")
        df = df.rename(columns={"bucket": "time"})
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        logger.error("load_ohlcv_nmin failed for %s @ %s: %s", symbol, resolution, e)
        return pd.DataFrame()


def load_ohlcv(symbol: str,
               resolution: str,
               from_date: str,
               to_date: str):
    if(resolution == 'D'):
        return load_ohlcv_day(symbol, from_date, to_date)
    elif(resolution == 'W'):
        return load_ohlcv_week(symbol, from_date, to_date)
    elif(resolution == 'M'):
        return load_ohlcv_month(symbol, from_date, to_date)
    elif(resolution == 'Y'):
        return load_ohlcv_year(symbol, from_date, to_date)
    else:
        try:
            res_int = int(resolution)
            if 1 <= res_int <= 240:
                return load_ohlcv_nmin(symbol, resolution + ' minutes', from_date, to_date)
        except ValueError:
            pass
        return None


def load_ohlcv_forex_15min(
    symbol: str,
    from_date: str,
    to_date: str,
    session_open_hour: int = 0,
    session_close_hour: int = 24,
) -> pd.DataFrame:
    """
    Load 1-minute forex data from the database and return it as-is for
    strategy use (strategies resample internally as needed).

    Forex data is stored in the same ``market_candles`` table by
    ``backend/data/fetch_forex.py`` with resolution ``'1'``.

    Parameters
    ----------
    symbol : str
        Forex pair as stored in the DB, e.g. ``"EURUSD"``.
    from_date : str
        Start timestamp with timezone, e.g. ``"2020-01-01 00:00:00+00:00"``.
    to_date : str
        End timestamp with timezone.
    session_open_hour : int
        UTC hour to start filtering bars (0 = no filter).
    session_close_hour : int
        UTC hour to stop filtering bars (24 = no filter).

    Returns
    -------
    pd.DataFrame
        Columns: time (UTC datetime), open, high, low, close, volume.
        Returns an empty DataFrame if no data is found.
    """
    engine = create_engine(DB_URL)
    sql = text("""
        SELECT time, open, high, low, close, volume
        FROM   market_candles
        WHERE  symbol     = :symbol
        AND    resolution = '1'
        AND    time >= :from_date
        AND    time <  :to_date
        ORDER  BY time
    """)

    df = pd.read_sql(sql, engine, params={
        "symbol":    symbol,
        "from_date": from_date,
        "to_date":   to_date,
    })

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Optional session filter (e.g. only London + NY hours)
    if session_open_hour > 0 or session_close_hour < 24:
        hour = df["time"].dt.hour
        df = df[
            (hour >= session_open_hour) & (hour < session_close_hour)
        ].reset_index(drop=True)

    return df


# ── Usage Examples ─────────────────────────────────────────────────────────
# from quandao_public.data.load import load_ohlcv, load_ohlcv_forex_15min
#
# # Indian equities (daily)
# df = load_ohlcv("NSE:NIFTY50-INDEX", "D", "2020-01-01 00:00:00+05:30", "2025-01-01 00:00:00+05:30")
#
# # Forex 1-min raw
# df = load_ohlcv_forex_15min("EURUSD", "2020-01-01 00:00:00+00:00", "2025-01-01 00:00:00+00:00")


