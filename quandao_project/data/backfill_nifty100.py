import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import psycopg2

# --- PROJECT ROOT SETUP ---
_file_path = Path(__file__).resolve()
_project_root = str(_file_path.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)

# Import existing fetch logic from quandao_project.data.fetch
from quandao_project.data.fetch import fetch_between, format_for_upsert, upsert_candles, DATABASE_URL, DAILY_START_DATE, IST

from quandao_project.data.universe import NIFTY_100_STOCKS

def main():
    print(f"\n{'='*50}")
    print(f"  NIFTY 100 DAILY BACKFILL START")
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    conn = psycopg2.connect(DATABASE_URL)
    total = len(NIFTY_100_STOCKS)
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    
    for i, symbol in enumerate(NIFTY_100_STOCKS, 1):
        print(f"[{i}/{total}] Backfilling {symbol} (Daily)...")
        try:
            # 1. Fetch data from Fyers
            df = fetch_between(symbol, DAILY_START_DATE, today_str, "D")
            
            if not df.empty:
                # 2. Format for DB
                formatted_rows = format_for_upsert(df, symbol, "D")
                # 3. Insert into DB
                upsert_candles(conn, formatted_rows)
                print(f"  [DB] Inserted {len(df)} rows for {symbol}")
            else:
                print(f"  [WARN] No data returned for {symbol}")
            
            # Rate limiting safety: Sleep briefly between stocks
            time.sleep(1) 
            
        except Exception as e:
            print(f"  [ERROR] Failed to fetch {symbol}: {e}")

    conn.close()
    print(f"\n{'='*50}")
    print(f"  BACKFILL COMPLETE")
    print(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
