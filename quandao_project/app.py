from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quandao_project.data.load import load_ohlcv

app = Flask(__name__)


def get_available_instruments():
    """Get list of available instruments from database"""
    return {
        "NIFTY 50": "NSE:NIFTY50-INDEX",
        "RELIANCE": "NSE:RELIANCE-EQ",
        "HDFCBANK": "NSE:HDFCBANK-EQ",
    }


def format_timestamp(dt):
    """Format datetime to required timestamp format for the DB/API"""
    return dt.strftime("%Y-%m-%d %H:%M:%S+05:30")


def get_time_delta_for_candles(resolution: str, num_candles: int) -> timedelta:
    """
    Calculate approximate time delta to fetch N candles of given resolution.
    Includes buffer for weekends, holidays, and market gaps.
    """
    if resolution == 'D':
        # Daily: ~1.5x for weekends/holidays
        return timedelta(days=int(num_candles * 1.5) + 5)
    elif resolution == 'W':
        # Weekly: ~1.2x buffer
        return timedelta(weeks=int(num_candles * 1.2) + 2)
    elif resolution == 'M':
        # Monthly: ~35 days per month
        return timedelta(days=num_candles * 35 + 30)
    else:
        # Intraday: resolution is minutes
        try:
            minutes = int(resolution)
            # Trading day = ~6.25 hours = 375 minutes
            # ~75 candles per day for 5-min, ~375 for 1-min
            candles_per_day = 375 / minutes
            days_needed = (num_candles / candles_per_day) * 2  # 2x buffer for gaps/weekends
            return timedelta(days=max(int(days_needed) + 3, 3))
        except ValueError:
            return timedelta(days=1000)  # Fallback



@app.route("/")
def index():
    """Main page with chart interface"""
    instruments = get_available_instruments()
    return render_template("index.html", instruments=instruments)


@app.route("/backtest")
def backtest_ui():
    """Backtest UI page with form inputs"""
    instruments = get_available_instruments()
    return render_template("backtest.html", instruments=instruments)


@app.route("/api/data")
def get_chart_data():
    """API endpoint to fetch OHLCV data with optional pagination for lazy loading"""
    try:
        symbol_name = request.args.get("symbol", "NIFTY 50")
        resolution = request.args.get("resolution", "D")
        days = int(request.args.get("days", 10000))
        from_param = request.args.get("from")
        to_param = request.args.get("to")

        # Pagination params for lazy loading
        limit = int(request.args.get("limit", 0))  # 0 = no limit (backwards compatible)
        offset_time = request.args.get("offset_time")  # ISO timestamp, load data before this

        instruments = get_available_instruments()
        db_symbol = instruments.get(symbol_name, "NSE:NIFTY50-INDEX")

        if from_param and to_param:
            from_datetime = datetime.fromisoformat(from_param)
            to_datetime = datetime.fromisoformat(to_param)
        else:
            to_datetime = datetime.now()
            # If offset_time is provided, use it as the upper bound
            if offset_time:
                to_datetime = datetime.fromisoformat(offset_time)

            # Use smart time range when limit is specified (lazy loading)
            if limit > 0:
                # Calculate time delta to fetch approximately 'limit' candles
                # Add extra buffer to ensure we get enough data
                smart_delta = get_time_delta_for_candles(resolution, int(limit * 1.3))
                from_datetime = to_datetime - smart_delta
            else:
                # No limit: use full days range (original behavior)
                from_datetime = to_datetime - timedelta(days=days)

        from_timestamp = format_timestamp(from_datetime)
        to_timestamp = format_timestamp(to_datetime)

        df = load_ohlcv(
            symbol=db_symbol,
            resolution=resolution,
            from_date=from_timestamp,
            to_date=to_timestamp,
        )

        if df.empty:
            return jsonify({"error": "No data available"}), 404

        if "time" not in df.columns:
            return jsonify({"error": 'No "time" column found in data'}), 500

        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        df = df.sort_values("time").reset_index(drop=True)

        # Track total available before limiting
        total_available = len(df)
        has_more = False
        earliest_time = None

        # Apply limit: take the most recent N candles
        # When using lazy loading (limit > 0), assume more data may exist unless we got no data
        # This handles cases where weekends/holidays cause gaps and we get less than limit
        if limit > 0:
            if len(df) > limit:
                # Definitely have more data
                has_more = True
                df = df.tail(limit).reset_index(drop=True)
            elif len(df) > 0:
                # Got some data but less than limit - there might still be more historical data
                # Only set hasMore=False if we explicitly know we've reached the beginning
                has_more = True  # Assume more exists; frontend will handle empty responses

        candlestick_data = []
        volume_data = []

        for _, row in df.iterrows():
            timestamp = int(row["time"].timestamp())
            candlestick_data.append(
                {
                    "time": timestamp,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
            volume_data.append(
                {
                    "time": timestamp,
                    "value": float(row["volume"]),
                    "color": "#1a5a54" if row["close"] >= row["open"] else "#7f312f",
                }
            )

        # Get earliest time for pagination cursor
        if len(df) > 0:
            earliest_time = int(df["time"].iloc[0].timestamp())

        current_price = float(df["close"].iloc[-1])
        first_price = float(df["close"].iloc[0])
        change = current_price - first_price
        change_pct = (change / first_price) * 100

        response_data = {
            "candlestickData": candlestick_data,
            "volumeData": volume_data,
            "stats": {
                "symbol": symbol_name,
                "resolution": resolution,
                "totalCandles": len(df),
                "totalAvailable": total_available,
                "currentPrice": current_price,
                "change": change,
                "changePct": change_pct,
                "high": float(df["high"].max()),
                "low": float(df["low"].min()),
                "volume": int(df["volume"].sum()),
            },
            "hasMore": has_more,
            "earliestTime": earliest_time,
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    """
    Backtest endpoint.
    Strategies are run via backend/strategies/*.py and their results
    are persisted to backend/results/output/.  Use GET /api/results
    (served by the frontend Node server) to view saved results in the UI.
    """
    return jsonify({
        "info": (
            "Run strategies via 'python -m backend.strategies.mp' and view "
            "results in the frontend dashboard at http://localhost:3000"
        )
    }), 501


@app.after_request
def add_header(response):
    """Prevent caching in the browser (always fetch fresh data)"""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
