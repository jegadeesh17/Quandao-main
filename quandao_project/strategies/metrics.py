"""
quandao_project/strategies/metrics.py
======================================
Shared performance metrics library — the single source of truth for ALL
risk-adjusted return calculations across every strategy in Quandao.

DESIGN PRINCIPLE:
    Every function is a "pure function":
        - Takes numpy arrays / pandas Series as input
        - Returns a single scalar or dict
        - No side effects, no global state, no file I/O
    
    This makes every function independently testable and explainable in interviews.

INTERVIEW ANSWERS EMBEDDED IN DOCSTRINGS:
    - Why Sortino over Sharpe? → Sortino only penalizes downside volatility.
    - Why Calmar? → Normalizes return by worst-case drawdown — better for intraday.
    - Why include zero-return days in Sharpe? → Prevents inflation by only counting trade days.

USAGE:
    from quandao_project.strategies.metrics import compute_all_metrics

    result = compute_all_metrics(trades=trade_list, daily_returns=daily_pnl_series)
    print(result['sharpe_ratio'], result['sortino_ratio'])
"""

import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series, list]


# ── Individual Metric Functions ─────────────────────────────────────────────

def sharpe_ratio(daily_returns: ArrayLike,
                 risk_free_rate: float = 0.065,
                 periods: int = 252) -> float:
    """
    Annualized Sharpe Ratio.

    Formula:
        Sharpe = (mean_daily_return - daily_rf) / std_daily_return * sqrt(periods)

    IMPORTANT: Pass ALL trading days (including days with no trade = 0 return).
    Passing only trade days inflates Sharpe by artificially reducing the sample size.

    Parameters
    ----------
    daily_returns : array-like  - Daily P&L values (in points, rupees, or %)
    risk_free_rate : float      - Annual risk-free rate (default 6.5% = India RBI)
    periods : int               - Trading days per year (252 for equities)

    Returns
    -------
    float : Annualized Sharpe ratio (0.0 if insufficient data or zero std)
    """
    r = np.array(daily_returns, dtype=float)
    if len(r) < 2:
        return 0.0

    daily_rf = risk_free_rate / periods
    excess_returns = r - daily_rf
    std = np.std(excess_returns, ddof=1)

    if std == 0:
        return 0.0
    return round(float((np.mean(excess_returns) / std) * np.sqrt(periods)), 4)


def sortino_ratio(daily_returns: ArrayLike,
                  risk_free_rate: float = 0.065,
                  periods: int = 252) -> float:
    """
    Annualized Sortino Ratio.

    WHY SORTINO OVER SHARPE:
        Sharpe penalizes BOTH upside and downside volatility equally.
        Sortino only penalizes DOWNSIDE volatility (the kind you actually care about).
        A strategy with high upside vol but low downside vol scores better on Sortino.

    Formula:
        Sortino = (mean_daily_return - daily_rf) / downside_std * sqrt(periods)
        downside_std = std of returns BELOW the risk-free rate (downside deviation)

    Parameters
    ----------
    daily_returns : array-like - Daily P&L or return values
    risk_free_rate : float     - Annual risk-free rate
    periods : int              - Trading days per year

    Returns
    -------
    float : Annualized Sortino ratio
    """
    r = np.array(daily_returns, dtype=float)
    if len(r) < 2:
        return 0.0

    daily_rf = risk_free_rate / periods
    excess_returns = r - daily_rf

    # Downside deviation: only returns BELOW the target (negative excess returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        # No losing days at all → perfect strategy → return a large number
        return 10.0

    downside_std = np.sqrt(np.mean(downside_returns ** 2))  # Root mean square of losses

    if downside_std == 0:
        return 0.0

    return round(float((np.mean(excess_returns) / downside_std) * np.sqrt(periods)), 4)


def calmar_ratio(daily_returns: ArrayLike, periods: int = 252) -> float:
    """
    Calmar Ratio.

    WHY CALMAR:
        Normalizes annualized return by the Maximum Drawdown.
        Better than Sharpe for strategies with fat tails or clustered losses,
        because it captures the worst-case scenario the strategy has experienced.
        Standard benchmark: Calmar > 1.0 is acceptable; > 3.0 is excellent.

    Formula:
        Calmar = Annualized Return / |Max Drawdown|

    Parameters
    ----------
    daily_returns : array-like - Daily P&L or return values
    periods : int              - Trading days per year

    Returns
    -------
    float : Calmar ratio (0.0 if no drawdown occurred, inf if no loss)
    """
    r = np.array(daily_returns, dtype=float)
    if len(r) < 2:
        return 0.0

    annualized_return = np.mean(r) * periods
    mdd = abs(max_drawdown(r))

    if mdd == 0:
        return float('inf') if annualized_return > 0 else 0.0

    return round(float(annualized_return / mdd), 4)


def max_drawdown(daily_returns: ArrayLike) -> float:
    """
    Maximum Drawdown (as a negative value).

    Formula:
        Build equity curve from cumulative sum.
        At each point: DD = (current_equity - running_peak) / running_peak
        Max DD = minimum of all DD values (most negative)

    WHY RETURN NEGATIVE:
        Drawdown IS a loss — it should be negative.
        Calmar uses |max_drawdown| to make the ratio positive.

    Parameters
    ----------
    daily_returns : array-like - Daily P&L values (in rupees or points)

    Returns
    -------
    float : Maximum drawdown (negative number, e.g. -0.15 = 15% drawdown)
    """
    r = np.array(daily_returns, dtype=float)
    if len(r) == 0:
        return 0.0

    equity_curve = np.cumsum(r)
    running_peak = np.maximum.accumulate(equity_curve)

    # Drawdown at each point: how far below the peak?
    drawdowns = equity_curve - running_peak

    return round(float(np.min(drawdowns)), 4)


def max_drawdown_duration(daily_returns: ArrayLike) -> int:
    """
    Maximum Drawdown Duration in days.

    Measures the longest period the strategy spent BELOW its previous peak.
    A high duration means the strategy is slow to recover — bad for capital efficiency.

    Returns
    -------
    int : Number of consecutive days in the deepest/longest drawdown period
    """
    r = np.array(daily_returns, dtype=float)
    if len(r) == 0:
        return 0

    equity_curve = np.cumsum(r)
    running_peak = np.maximum.accumulate(equity_curve)

    in_drawdown = equity_curve < running_peak

    max_duration = 0
    current_duration = 0

    for dd in in_drawdown:
        if dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def win_rate(trades: list) -> float:
    """
    Win rate: fraction of trades with positive P&L.

    Parameters
    ----------
    trades : list of dicts, each must have a 'pnl' key.

    Returns
    -------
    float : Win rate in [0, 1]
    """
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t.get('pnl', 0) > 0)
    return round(winners / len(trades), 4)


def profit_factor(trades: list) -> float:
    """
    Profit Factor = Gross Profit / Gross Loss.

    Interpretation:
        > 1.5 → acceptable
        > 2.0 → good
        < 1.0 → losing strategy (guaranteed eventual ruin)

    Parameters
    ----------
    trades : list of dicts with 'pnl' key.

    Returns
    -------
    float : Profit factor (inf if no losing trades)
    """
    gross_profit = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
    gross_loss   = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return round(gross_profit / gross_loss, 4)


def value_at_risk(daily_returns: ArrayLike, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk (VaR) at the given confidence level.

    Formula:
        VaR = percentile of returns at (1 - confidence) level
        e.g. VaR_95 = 5th percentile of daily returns

    Interpretation:
        VaR(95%) = -X means: "On 95% of days, the loss will not exceed X."

    Parameters
    ----------
    daily_returns : array-like - Daily P&L values
    confidence : float         - Confidence level (0.95 = 95%)

    Returns
    -------
    float : VaR value (negative number represents a loss threshold)
    """
    r = np.array(daily_returns, dtype=float)
    if len(r) == 0:
        return 0.0

    return round(float(np.percentile(r, (1 - confidence) * 100)), 4)


def conditional_var(daily_returns: ArrayLike, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall.

    CVaR is BETTER than VaR because it answers:
        "Given that we ARE in the worst (1-confidence)% of days, what is the
         average loss?" — captures the tail risk beyond VaR.

    Formula:
        CVaR = mean(returns where returns <= VaR)

    Parameters
    ----------
    daily_returns : array-like - Daily P&L values
    confidence : float         - Confidence level

    Returns
    -------
    float : CVaR (negative number, worse than VaR)
    """
    r = np.array(daily_returns, dtype=float)
    if len(r) == 0:
        return 0.0

    var = value_at_risk(r, confidence)
    tail_returns = r[r <= var]

    if len(tail_returns) == 0:
        return var

    return round(float(np.mean(tail_returns)), 4)


def recovery_factor(daily_returns: ArrayLike) -> float:
    """
    Recovery Factor = Total Net P&L / |Max Drawdown|.

    Measures how many times the strategy "earned back" its worst drawdown.
    Higher is better. A recovery factor < 1 means the strategy never recovered
    from its worst loss on a net basis — a serious red flag.
    """
    r = np.array(daily_returns, dtype=float)
    total_pnl = float(np.sum(r))
    mdd = abs(max_drawdown(r))

    if mdd == 0:
        return float('inf') if total_pnl > 0 else 0.0

    return round(total_pnl / mdd, 4)


# ── Master Aggregation Function ─────────────────────────────────────────────

def compute_all_metrics(trades: list,
                        all_trading_dates: list = None,
                        risk_free_rate: float = 0.065) -> dict:
    """
    Compute ALL performance metrics in one call.

    CRITICAL: This function properly includes all trading dates (including
    no-trade days as zero-return) when computing ratio metrics. This prevents
    inflated Sharpe ratios from only counting active trade days.

    Parameters
    ----------
    trades : list of dicts
        Each dict must have: {'date': 'YYYY-MM-DD', 'pnl': float, ...}

    all_trading_dates : list of date objects or strings, optional
        All trading dates in the backtest period (including no-trade days).
        If None, only trade days are used (less accurate, but acceptable).

    risk_free_rate : float
        Annual risk-free rate for ratio computations.

    Returns
    -------
    dict : Complete metrics dictionary with all ratios, drawdown stats, and trade counts.
    """
    if not trades:
        return _empty_metrics()

    # Build daily P&L map: date → total P&L on that day
    daily_pnl_map = {}
    for t in trades:
        date_key = str(t.get('date', ''))
        daily_pnl_map[date_key] = daily_pnl_map.get(date_key, 0.0) + float(t.get('pnl', 0.0))

    # Build daily returns series (include zero-return days for correct Sharpe)
    if all_trading_dates:
        daily_returns = [daily_pnl_map.get(str(d), 0.0) for d in all_trading_dates]
    else:
        daily_returns = list(daily_pnl_map.values())

    daily_returns_arr = np.array(daily_returns, dtype=float)

    # Trade-level stats
    pnl_values = [t.get('pnl', 0.0) for t in trades]
    wins   = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]

    total_pnl   = sum(pnl_values)
    mdd         = max_drawdown(daily_returns_arr)
    mdd_days    = max_drawdown_duration(daily_returns_arr)

    return {
        # Trade counts
        'trades_executed':           len(trades),
        'winning_trades':            len(wins),
        'losing_trades':             len(losses),
        'breakeven_trades':          len(trades) - len(wins) - len(losses),

        # P&L summary
        'total_pnl':                 round(total_pnl, 2),
        'avg_win':                   round(np.mean(wins), 2)   if wins   else 0.0,
        'avg_loss':                  round(np.mean(losses), 2) if losses else 0.0,
        'max_win':                   round(max(wins), 2)       if wins   else 0.0,
        'max_loss':                  round(min(losses), 2)     if losses else 0.0,

        # Win/Loss stats
        'win_rate':                  round(len(wins) / len(trades), 4) if trades else 0.0,
        'profit_factor':             profit_factor(trades),

        # Risk-adjusted ratios
        'sharpe_ratio':              sharpe_ratio(daily_returns_arr, risk_free_rate),
        'sortino_ratio':             sortino_ratio(daily_returns_arr, risk_free_rate),
        'calmar_ratio':              calmar_ratio(daily_returns_arr),
        'recovery_factor':           recovery_factor(daily_returns_arr),

        # Drawdown
        'max_drawdown':              round(mdd, 2),
        'max_drawdown_duration_days': mdd_days,

        # Risk metrics
        'var_95':                    value_at_risk(daily_returns_arr, 0.95),
        'cvar_95':                   conditional_var(daily_returns_arr, 0.95),
    }


def _empty_metrics() -> dict:
    """Return a zeroed metrics dict for when there are no trades."""
    return {k: 0.0 for k in [
        'trades_executed', 'winning_trades', 'losing_trades', 'breakeven_trades',
        'total_pnl', 'avg_win', 'avg_loss', 'max_win', 'max_loss',
        'win_rate', 'profit_factor', 'sharpe_ratio', 'sortino_ratio',
        'calmar_ratio', 'recovery_factor', 'max_drawdown',
        'max_drawdown_duration_days', 'var_95', 'cvar_95',
    ]}
