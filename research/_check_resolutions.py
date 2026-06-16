import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
import psycopg2

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()
cur.execute("""
    SELECT symbol, resolution, count(*)
    FROM market_candles
    WHERE symbol IN ('NSE:NIFTY50-INDEX', 'NSE:RELIANCE-EQ', 'NSE:HDFCBANK-EQ', 'NSE:GOLDBEES-EQ')
    GROUP BY symbol, resolution
    ORDER BY symbol, resolution
""")
for row in cur.fetchall():
    print(f"  {row[0]:25s}  res={row[1]:5s}  rows={row[2]}")
conn.close()
