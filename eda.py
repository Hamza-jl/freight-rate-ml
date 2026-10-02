"""Generate the figures used in the report (run after train.py).

    python eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from freight_rate import config  # noqa: E402
from freight_rate.data import load_clean_training  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 160,
    }
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / name, bbox_inches="tight")
    plt.close(fig)


def market_vs_rates(df):
    """market_index swings far more than rates do -> it is not a reliable level signal."""
    clean = df[~df["is_corrupted"]].assign(rpm=lambda d: d[config.TARGET] / d["distance"])
    monthly = clean.groupby(clean["date"].dt.to_period("M")).agg(rpm=("rpm", "median"), mi=("market_index", "median"))
    indexed = 100 * monthly / monthly.iloc[0]
    x = indexed.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(x, indexed["mi"], color=ORANGE, lw=2, marker="o", ms=5, label="market_index (median)")
    ax.plot(x, indexed["rpm"], color=BLUE, lw=2, marker="o", ms=5, label="Rate per mile (median)")
    ax.set_ylabel("Index (Jan 2025 = 100)")
    ax.set_title("market_index swings ~40%; actual rates move ~10%")
    ax.legend(loc="upper left")
    save(fig, "market_vs_rates.png")


def corrupted_labels(df):
    clean = df[~df["is_corrupted"]]
    rpm_med = clean.groupby("equipment").apply(lambda g: (g[config.TARGET] / g["distance"]).median())
    expected = df["distance"] * df["equipment"].map(rpm_med)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ok, bad = ~df["is_corrupted"], df["is_corrupted"]
    ax.scatter(expected[ok], df.loc[ok, config.TARGET], s=3, color=NEUTRAL, alpha=0.4, lw=0, label="Kept")
    ax.scatter(expected[bad], df.loc[bad, config.TARGET], s=10, color=ORANGE, alpha=0.9, lw=0,
               label=f"Flagged as corrupted ({bad.sum()})")
    ticks = [100, 300, 1000, 3000, 10000]
    ax.set_xscale("log"); ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Expected rate ($): distance x median rate/mile for equipment")
    ax.set_ylabel("posted_rate ($)")
    ax.grid(axis="x", color=GRID)
    ax.set_title("1.4% of labels are off by 2-5x")
    ax.legend(loc="upper left", markerscale=2)
    save(fig, "corrupted_labels.png")


def daily_market(df):
    daily = df.groupby("date")["market_index"].median()
    val = pd.read_csv(config.VALIDATION_PATH, parse_dates=["date"]).groupby("date")["market_index"].median()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(daily.index, daily, color=BLUE, lw=1.2, label="train_test.csv")
    ax.plot(val.index, val, color=AQUA, lw=1.2, label="validation.csv")
    ax.set_ylabel("Daily median market_index")
    ax.set_title("market_index is a daily market value with a 7-day cycle (R² = 0.98 on date alone)")
    ax.legend(loc="upper left")
    save(fig, "daily_market_index.png")


def backtest_chart():
    bt = pd.read_csv(config.REPORT_DIR / "backtest.csv")
    models = list(dict.fromkeys(bt["model"]))
    folds = list(dict.fromkeys(bt["fold"]))
    colors = [NEUTRAL, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.26
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = bt[bt["model"] == model].set_index("fold").loc[folds, "MAPE"]
        bars = ax.bar(np.arange(len(folds)) + (i - 1) * width, vals, width - 0.03, color=color, label=model)
        ax.bar_label(bars, fmt="%.2f%%", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(np.arange(len(folds)), [f.replace(" -> ", "\n-> ") for f in folds])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Rolling-origin backtest, 2-month horizon (lower is better)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, bt["MAPE"].max() * 1.35)
    save(fig, "backtest_mape.png")


def importance_chart():
    imp = pd.Series(json.loads(config.METRICS_PATH.read_text())["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(imp.index, 100 * imp.values, color=BLUE, height=0.6)
    for y, v in enumerate(100 * imp.values):
        ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("Feature importance, final model")
    save(fig, "feature_importance.png")


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = load_clean_training()
    market_vs_rates(df)
    corrupted_labels(df)
    daily_market(df)
    backtest_chart()
    importance_chart()
    print(f"Figures written to {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
