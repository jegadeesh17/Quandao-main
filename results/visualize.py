"""
backend/results/visualize.py
Chart & PDF generator for strategy results.

Usage:
    from quandao_public.results.visualize import generate_report
    generate_report("backend/results/output/moon_phase_2025.json")
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)

# Output directory for charts (sibling to this file)
_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal chart helpers
# ---------------------------------------------------------------------------

def _save_moon_phase_bar(moon_phase: dict, prefix: str) -> str:
    path = str(_OUTPUT_DIR / f"{prefix}_moon_phase_break_pct.png")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        ["Full Moon Break Week High %", "New Moon Break Week Low %"],
        [moon_phase["full_moon_periods"]["break_week_high_pct"],
         moon_phase["new_moon_periods"]["break_week_low_pct"]],
        color=["blue", "orange"],
    )
    plt.title("Moon Phase Break Percentages")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval}%",
                 ha="center", va="bottom")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _save_gap_analysis_bar(gap_analysis: dict, prefix: str) -> str:
    path = str(_OUTPUT_DIR / f"{prefix}_gap_analysis.png")
    gap_types = list(gap_analysis.keys())
    full_moon_counts = [gap_analysis[gt]["full_moon_periods"]["break_week_high_count"] for gt in gap_types]
    new_moon_counts  = [gap_analysis[gt]["new_moon_periods"]["break_week_low_count"]  for gt in gap_types]
    width = 0.35
    x = range(len(gap_types))
    plt.figure(figsize=(12, 7))
    plt.bar(x, full_moon_counts, width, label="Full Moon Break High", color="blue")
    plt.bar([p + width for p in x], new_moon_counts, width, label="New Moon Break Low", color="orange")
    plt.xlabel("Gap Type"); plt.ylabel("Count")
    plt.title("Gap Analysis by Moon Phase (Break Counts)")
    plt.xticks([p + width / 2 for p in x], gap_types)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


def _save_win_loss_pie(perf: dict, prefix: str) -> str:
    path = str(_OUTPUT_DIR / f"{prefix}_win_loss_pie.png")
    plt.figure(figsize=(8, 8))
    plt.pie(
        [perf["winning_trades"], perf["losing_trades"]],
        labels=["Winning Trades", "Losing Trades"],
        colors=["green", "red"],
        explode=(0.1, 0),
        autopct="%1.1f%%", shadow=True, startangle=140,
    )
    plt.title(f"Win Rate: {perf['win_rate']}%")
    plt.axis("equal")
    plt.savefig(path); plt.close()
    return path


def _save_exit_reasons_pie(perf: dict, prefix: str) -> str:
    path = str(_OUTPUT_DIR / f"{prefix}_exit_reasons.png")
    exit_reasons = perf["exit_reasons"]
    plt.figure(figsize=(8, 8))
    plt.pie(list(exit_reasons.values()), labels=list(exit_reasons.keys()),
            autopct="%1.1f%%", shadow=True, startangle=90)
    plt.title("Exit Reasons Distribution")
    plt.axis("equal")
    plt.savefig(path); plt.close()
    return path


def _save_cumulative_pnl(trades_df: pd.DataFrame, prefix: str) -> str:
    path = str(_OUTPUT_DIR / f"{prefix}_cumulative_pnl.png")
    executed = trades_df[trades_df["trade_taken"] == True].copy() if "trade_taken" in trades_df.columns else trades_df.copy()
    executed["cumulative_pnl"] = executed["pnl"].cumsum()
    plt.figure(figsize=(12, 7))
    plt.plot(executed["date"], executed["cumulative_pnl"], marker="o", linestyle="-",
             color="purple", label="Cumulative PnL")
    plt.title("Cumulative PnL Over Trades"); plt.xlabel("Date"); plt.ylabel("Cumulative PnL")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.legend(); plt.tight_layout()
    plt.savefig(path); plt.close()
    return path


def _save_pnl_per_trade(trades_df: pd.DataFrame, prefix: str) -> str:
    path = str(_OUTPUT_DIR / f"{prefix}_pnl_per_trade.png")
    executed = trades_df[trades_df["trade_taken"] == True].copy() if "trade_taken" in trades_df.columns else trades_df.copy()
    colors = ["green" if p > 0 else "red" for p in executed["pnl"]]
    plt.figure(figsize=(12, 7))
    plt.bar(executed["date"].dt.strftime("%Y-%m-%d"), executed["pnl"], color=colors)
    plt.title("PnL per Trade"); plt.xlabel("Date"); plt.ylabel("PnL")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(path); plt.close()
    return path


def _save_key_metrics_table(perf: dict, prefix: str) -> str:
    path = str(_OUTPUT_DIR / f"{prefix}_metrics_table.png")
    metrics_rows = [
        ["Total Opportunities",        perf.get("total_opportunities", "N/A")],
        ["Trades Executed",            perf.get("trades_executed", "N/A")],
        ["Execution Rate",             f"{perf.get('execution_rate', 0):.1f}%"],
        ["Win Rate",                   f"{perf.get('win_rate', 0):.1f}%"],
        ["Total PnL",                  f"{perf.get('total_pnl', 0):.2f}"],
        ["Avg Win",                    f"{perf.get('avg_win', 0):.2f}"],
        ["Avg Loss",                   f"{perf.get('avg_loss', 0):.2f}"],
        ["Sharpe Ratio",               f"{perf.get('sharpe_ratio', 0):.2f}"],
        ["Profit Factor",              str(perf.get("profit_factor", "N/A"))],
        ["Max Drawdown",               f"{perf.get('max_drawdown', 0):.2f}"],
        ["Avg Trade Duration (min)",   f"{perf.get('avg_trade_duration_minutes', 0):.2f}"],
        ["Recovery Factor",            str(perf.get("recovery_factor", "N/A"))],
        ["Win / Loss Ratio",           str(perf.get("win_loss_ratio", "N/A"))],
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    tbl = ax.table(cellText=metrics_rows, colLabels=["Metric", "Value"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2, 1.5)
    for j in range(2):
        tbl[(0, j)].set_facecolor("#4CAF50")
    plt.title("Key Trading Performance Metrics", y=0.95, fontsize=14, fontweight="bold")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    return path


def _save_yearly_performance(yearly_perf: dict, prefix: str) -> str:
    path = str(_OUTPUT_DIR / f"{prefix}_yearly.png")
    years = sorted(yearly_perf.keys())
    pnls  = [yearly_perf[y]["total_pnl"] for y in years]
    colors = ["green" if p > 0 else "red" for p in pnls]
    plt.figure(figsize=(14, 7))
    bars = plt.bar(years, pnls, color=colors, alpha=0.7, edgecolor="black")
    plt.xlabel("Year"); plt.ylabel("Total PnL")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for i, (year, pnl) in enumerate(zip(years, pnls)):
        perf = yearly_perf[year]
        plt.text(i, pnl + (max(pnls) * 0.02 if pnl > 0 else -abs(max(pnls)) * 0.02),
                 f"{pnl:.2f}\n{perf.get('trades_executed',0)} trades\n{perf.get('win_rate',0)}% WR",
                 ha="center", va="bottom" if pnl > 0 else "top", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path); plt.close()
    return path


def _save_monthly_heatmap(monthly_perf: dict, prefix: str) -> str:
    path = str(_OUTPUT_DIR / f"{prefix}_monthly_heatmap.png")
    monthly_data: dict = {}
    for ym, perf in monthly_perf.items():
        year, month = ym.split("-")
        monthly_data.setdefault(year, {})[int(month)] = perf["total_pnl"]
    years = sorted(monthly_data.keys())
    heatmap = np.array([[monthly_data[y].get(m, 0) for m in range(1, 13)] for y in years])
    vmin, vmax = heatmap.min(), heatmap.max()
    plt.figure(figsize=(16, 10))
    if vmin == vmax:
        im = plt.imshow(heatmap, cmap="RdYlGn", aspect="auto")
    elif vmin >= 0:
        im = plt.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
    elif vmax <= 0:
        im = plt.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = plt.imshow(heatmap, cmap="RdYlGn", aspect="auto", norm=norm)
    plt.xticks(range(12), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.yticks(range(len(years)), years)
    cbar = plt.colorbar(im); cbar.set_label("PnL", rotation=270, labelpad=20)
    for i, year in enumerate(years):
        for j, month in enumerate(range(1, 13)):
            pnl = monthly_data[year].get(month, 0)
            if pnl != 0:
                tc = "white" if abs(pnl) > abs(heatmap.max()) * 0.5 else "black"
                plt.text(j, i, f"{pnl:.1f}", ha="center", va="center",
                         color=tc, fontsize=8, fontweight="bold")
    plt.title("Monthly Performance Heatmap (PnL)", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Month"); plt.ylabel("Year"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(json_path: str, prefix: str | None = None) -> str:
    """
    Generate a full PDF report from a strategy result JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to the strategy result JSON (produced by ``report.save_json``).
    prefix : str, optional
        Prefix for chart image files. Defaults to the JSON file stem.

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """
    json_path = Path(json_path)
    prefix = prefix or json_path.stem

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata     = data.get("metadata", {})
    perf_section = data.get("performance_analysis", data)
    moon_phase   = perf_section.get("moon_phase_analysis", {})
    gap_analysis = perf_section.get("gap_analysis_by_moon_phase", {})
    trading_perf = perf_section.get("trading_performance", {})
    yearly_perf  = perf_section.get("yearly_performance", {})
    monthly_perf = perf_section.get("monthly_performance", {})

    trades_raw = data.get("detailed_trades", data.get("trades", []))
    trades_df  = pd.DataFrame(trades_raw)
    if len(trades_df) > 0:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df = trades_df.sort_values("date")

    # --- Generate chart PNGs ---
    images = []
    if moon_phase:
        images.append((_save_moon_phase_bar(moon_phase, prefix),   "Moon Phase Break Percentages"))
    if gap_analysis:
        images.append((_save_gap_analysis_bar(gap_analysis, prefix), "Gap Analysis by Moon Phase"))
    if trading_perf.get("winning_trades") is not None:
        images.append((_save_win_loss_pie(trading_perf, prefix),   "Win/Loss Distribution"))
    if trading_perf.get("exit_reasons"):
        images.append((_save_exit_reasons_pie(trading_perf, prefix), "Exit Reasons"))
    if len(trades_df) > 0 and "pnl" in trades_df.columns:
        images.append((_save_cumulative_pnl(trades_df, prefix),    "Cumulative PnL Over Time"))
        images.append((_save_pnl_per_trade(trades_df, prefix),     "PnL per Trade"))
    if trading_perf:
        images.append((_save_key_metrics_table(trading_perf, prefix), "Key Metrics Summary"))
    if yearly_perf:
        images.append((_save_yearly_performance(yearly_perf, prefix), "Yearly Performance"))
    if monthly_perf:
        images.append((_save_monthly_heatmap(monthly_perf, prefix),   "Monthly Performance Heatmap"))

    # --- Build PDF ---
    pdf_path = str(_OUTPUT_DIR / f"{prefix}_report.pdf")
    doc   = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("TitleStyle", fontSize=20, alignment=1,
                                 spaceAfter=20, leading=24)
    strategy_name = metadata.get("strategy_name", prefix.replace("_", " ").title())
    story.append(Paragraph(f"{strategy_name} – Strategy Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 12))

    if metadata:
        desc = (
            f"<b>Strategy:</b> {metadata.get('strategy_description','N/A')}<br/><br/>"
            f"<b>Period:</b> {metadata.get('daily_records_analyzed', '?')} daily records<br/>"
            f"<b>Stop Loss:</b> {metadata.get('stop_loss_points', '?')} points<br/>"
            f"<b>Gap Threshold:</b> {metadata.get('gap_threshold_pct', '?')}%"
        )
        story.append(Paragraph(desc, styles["Normal"]))
        story.append(Spacer(1, 24))

    for img_path, caption in images:
        try:
            story.append(Image(img_path, width=6.5 * inch, height=4.5 * inch))
            story.append(Paragraph(f"<b>{caption}</b>", styles["Heading3"]))
            story.append(Spacer(1, 12))
        except Exception as e:
            print(f"[visualize] Warning: could not add {img_path}: {e}")

    doc.build(story)
    print(f"[results] PDF report saved → {pdf_path}")
    return pdf_path

