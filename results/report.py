"""
backend/results/report.py
Standardised result-saving utility for all Quandao strategies.

Usage from a strategy:
    from quandao_public.results.report import save_json, save_summary

    result = my_strategy.run(df, ...)
    save_json(result, "moon_phase_2025")
    save_summary(result["metrics"], "moon_phase_2025")
"""

import json
import textwrap
from datetime import datetime
from pathlib import Path

# All output files land in quandao_private/results/backtest/
_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, filename: str) -> Path:
    """
    Save a strategy result dict to ``quandao_private/results/backtest/<filename>.json``.

    Parameters
    ----------
    data : dict
        Full result dict returned by a strategy's ``run()`` method.
        Expected keys: ``trades``, ``metrics``, and optionally
        ``metadata``, ``yearly_performance``, ``monthly_performance``.
    filename : str
        Base name without extension, e.g. ``"moon_phase_2025"``.

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    out_path = _OUTPUT_DIR / f"{filename}.json"
    
    # Enforce standard JSON structure
    standardized_data = {
        "metadata": data.get("metadata", {}),
        "metrics": data.get("metrics", {}),
        "trades": data.get("trades", [])
    }
    
    # Attach a generation timestamp
    standardized_data["_generated_at"] = datetime.now().isoformat()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(standardized_data, f, indent=2, default=str)
    print(f"[results] JSON saved -> {out_path}")
    return out_path


def save_summary(metrics: dict, filename: str) -> Path:
    """
    Save a human-readable text summary of strategy metrics.

    Parameters
    ----------
    metrics : dict
        Dict with at minimum the keys produced by
        ``analyze_strategy_performance()`` or similar metric calculators.
    filename : str
        Base name without extension, e.g. ``"moon_phase_2025"``.

    Returns
    -------
    Path
        Absolute path of the saved ``.txt`` file.
    """
    out_path = _OUTPUT_DIR / f"{filename}_summary.txt"

    lines = [
        "=" * 60,
        f"  Strategy Summary: {filename}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    # Standard metrics matching intraday_orb_forex_summary.txt layout
    ordered_keys = [
        ("total_opportunities", "Total Opportunities"),
        ("trades_executed", "Trades Executed"),
        ("win_rate", "Win Rate (%)"),
        ("total_pnl", "Total PnL"),
        ("avg_win", "Avg Win"),
        ("avg_loss", "Avg Loss"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("max_drawdown", "Max Drawdown"),
        ("profit_factor", "Profit Factor"),
        ("winning_trades", "winning_trades"),
        ("losing_trades", "losing_trades"),
    ]

    for key, label in ordered_keys:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                lines.append(f"  {label:<38} {val:.5f}")
            else:
                lines.append(f"  {label:<38} {val}")
                
    # Handle the pips/points keys specifically
    # Expectancy
    if "expectancy_pips" in metrics:
        lines.append(f"  {'expectancy_pips':<38} {metrics['expectancy_pips']}")
    elif "expectancy_points" in metrics:
        lines.append(f"  {'expectancy_points':<38} {metrics['expectancy_points']}")
        
    # Sortino
    if "sortino_ratio" in metrics:
        lines.append(f"  {'sortino_ratio':<38} {metrics['sortino_ratio']}")
        
    # Calmar
    if "calmar_ratio" in metrics:
        lines.append(f"  {'calmar_ratio':<38} {metrics['calmar_ratio']}")
        
    # MAE
    if "avg_mae_pips" in metrics:
        lines.append(f"  {'avg_mae_pips':<38} {metrics['avg_mae_pips']}")
    elif "avg_mae_points" in metrics:
        lines.append(f"  {'avg_mae_points':<38} {metrics['avg_mae_points']}")
        
    # MFE
    if "avg_mfe_pips" in metrics:
        lines.append(f"  {'avg_mfe_pips':<38} {metrics['avg_mfe_pips']}")
    elif "avg_mfe_points" in metrics:
        lines.append(f"  {'avg_mfe_points':<38} {metrics['avg_mfe_points']}")

    lines += ["", "=" * 60]
    text = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[results] Summary saved -> {out_path}")
    return out_path

