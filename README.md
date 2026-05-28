# Quandao Trading System — Client Operations Guide (A to Z)

Welcome to the **Quandao Trading System**. This system is an automated algorithmic trading application designed to trade NIFTY50 index options and futures based on a combination of opening range breakouts and astronomical moon-phase filters.

This guide is designed for clients and administrators from both non-technical and technical backgrounds to easily set up, run, and monitor the trading bot.

---

## Table of Contents
1. **Understanding the Strategy (The "Why")**
2. **Prerequisites & System Setup**
3. **Configuration & Credentials (The `.env` File)**
4. **Operation Steps**
   * Running Historical Backtests (Simulations)
   * Running in Dry-Run Mode (Paper Trading / Safe Testing)
   * Running in Live Mode (Real Money Broker Execution)
5. **Monitoring & Checking Status**
6. **Interactive Dashboard Guide (`guide.ipynb`)**

---

## 1. Understanding the Strategy (The "Why")

The system runs a **Moon-Phase Opening Range Breakout (ORB)** model. It operates on two layers of filters before deciding to take a trade:

1.  **Layer 1: Directional Moon Bias (Astronomical Filter)**
    *   **New Moon Window**: For 14 days after a New Moon, the system has a **LONG (Buy-only)** bias.
    *   **Full Moon Window**: For 14 days after a Full Moon, the system has a **SHORT (Sell-only)** bias.
    *   If there is no clear moon phase or the windows overlap, the bot skips trading for the day.
2.  **Layer 2: Breakout Entry (Technical Filter)**
    *   Every morning, the bot records the highest and lowest prices of NIFTY50 during the first **10 minutes** of the stock market session (09:15 to 09:25 IST).
    *   **LONG Entry**: If the Moon Bias is Buy-only, the bot waits for the price to break *above* the 10-minute High (provided the price is also above the day's Volume-Weighted Average Price, or VWAP).
    *   **SHORT Entry**: If the Moon Bias is Sell-only, the bot waits for the price to break *below* the 10-minute Low (provided the price is also below the day's VWAP).
3.  **Automatic Exit & Risk Management**
    *   **Stop Loss**: If the trade goes against us, the bot automatically exits the position using a dynamic Stop Loss calculated via Average True Range (ATR) volatility.
    *   **Max 1 Trade/Day**: The bot only executes at most one trade per day to limit drawdown.
    *   **End-of-Day Exit (15:15 IST)**: If neither the Stop Loss nor a Time Limit is hit, the bot automatically closes any open position at 15:15 IST, ensuring no trades are carried overnight.

---

## 2. Prerequisites & System Setup

To run this system, you need a standard Windows computer with Python and database tools installed.

### Step 2.1: Install Python (Version 3.10+)
1. Download Python from the [official website](https://www.python.org/downloads/).
2. Run the installer and **crucially**, check the box that says **"Add Python to PATH"** before clicking Install.

### Step 2.2: Open Terminal / Command Prompt
* Press `Windows Key + R`, type `cmd`, and press Enter.
* Use the terminal to navigate to the directory where this project is located:
  ```cmd
  cd C:\Users\jegad\Quandao-main
  ```

### Step 2.3: Install System Dependencies
Run this command in the terminal to automatically install the libraries required to run the bot:
```bash
pip install -r requirements.txt
```

---

## 3. Configuration & Credentials (The `.env` File)

The system stores your account details and database passwords securely in a file called `.env` in the root folder of the project.

Open the `.env` file in Notepad and customize the following settings:

```env
# 1. Database Connection
DB_USER=postgres
DB_PASSWORD=your_postgres_password
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=quandao

# 2. Broker Credentials (Fyers API)
FYERS_APP_ID=your_fyers_app_id
FYERS_SECRET_KEY=your_fyers_secret_key
FYERS_ACCESS_TOKEN=your_current_fyers_access_token
```

> [!TIP]
> **Refreshing Fyers Access Tokens**: Fyers access tokens expire daily. To log in and refresh your token, run:
> ```bash
> python -m quandao_project.data.data_fetcher --source fyers --mode login
> ```

---

## 4. Operation Steps

### 4.1 Running Historical Backtests (Simulations)
If you want to simulate how the strategy would have performed over the past year (June 2025 – April 2026), execute:
```bash
python -m quandao_project.main --backtest
```
This prints the stats directly to your screen and creates:
*   `moonphase_orb.json`: Detailed list of every simulated trade.
*   `moonphase_orb_summary.json`: Strategy performance metrics (win%, drawdown, total profit).

### 4.2 Running in Dry-Run Mode (Paper Trading)
To run the bot in real-time during live market hours (09:15 to 15:30 IST) without placing real money at risk:
```bash
python -m quandao_project.strategies.moonphase_orb --dry-run
```
*   The bot will pull live prices, compute boundaries, and simulate entries and exits.
*   Simulated trades are logged in `fyers_paper_trades.json` in the root directory.

### 4.3 Running in Live Mode (Real Execution)
To start the bot and route actual orders to your Fyers trading account:
```bash
python -m quandao_project.strategies.moonphase_orb
```
> [!WARNING]
> Ensure that you have updated your daily access token and have sufficient margin in your broker account before activating Live execution.

---

## 5. Monitoring & Checking Status

*   **Heartbeats**: When running live or in dry-run mode, the bot outputs a `HEARTBEAT` log line every second to show it is active, showing the current price, moon bias, and position state.
*   **State Recovery File (`moonphase_orb_state.json`)**: If the bot is closed or crashes mid-day, restarting it is completely safe. The bot reads this state file to instantly remember if it already took a trade, what the daily high/low boundaries are, and any active position details.
*   **Logs**: Check `moonphase_orb_live.log` in the folder for permanent audit trails.

---

## 6. Interactive Dashboard Guide (`guide.ipynb`)

If you prefer a graphical user interface over the command line, we have provided a Jupyter Notebook dashboard called `guide.ipynb` in the project root.

To open the dashboard:
1.  Run the command:
    ```bash
    jupyter notebook guide.ipynb
    ```
2.  Your web browser will open.
3.  You can execute cells sequentially to:
    *   **Run a Backtest** with a click.
    *   **View dynamic tables** summarizing the strategy returns.
    *   **Plot the cumulative PnL curve** to visualize profits.
    *   **Run the live bot dry-run** directly from the notebook.
