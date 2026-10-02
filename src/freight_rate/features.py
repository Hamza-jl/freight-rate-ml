"""Feature engineering.

Key idea: market_index and quote_signal each mix two things.
  1. A day-level market regime, shared by every load on that date.
  2. A load-level deviation from that regime.

The day-level market_index swings 0.82-1.30 across the year (a weekly cycle
plus a spring/summer hump), but rates only move a few percent with it, and its
level does not map to the same rate level in different months. Trees fed the raw
value learn month-specific splits that do not carry over to new months
(backtest MAPE 3.4% with it vs 2.6% without). Only the load-level deviation is kept.

quote_signal is kept raw (its daily level separates regime months) plus its
load-level deviation, which carries most of its per-load signal.

Day-level stats are computed from the frame being scored. That is legitimate:
validation.csv provides the feature rows for every load on each date, and no
labels are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "distance",
    "weight",
    "equipment",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "day_of_week",
    "quote_signal",
    "quote_signal_dev",
    "market_index_dev",
]


def build_features(df: pd.DataFrame, market_stats: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["market_index_day", "quote_signal_day"], errors="ignore")
    df = df.merge(market_stats, on="date", how="left")
    market_index = df["market_index"].fillna(df["market_index_day"])
    out = pd.DataFrame(index=df.index)
    out["distance"] = df["distance"]
    out["weight"] = df["weight"]
    out["equipment"] = pd.Categorical(df["equipment"], categories=["Dry Van", "Flatbed", "Reefer"])
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        out[col] = df[col]
    out["day_of_week"] = df["date"].dt.dayofweek
    out["quote_signal"] = df["quote_signal"]
    out["quote_signal_dev"] = df["quote_signal"] - df["quote_signal_day"]
    out["market_index_dev"] = market_index - df["market_index_day"]
    return out[FEATURES]


def log_target(rate: pd.Series) -> pd.Series:
    return np.log(rate)


def inverse_target(pred) -> np.ndarray:
    return np.exp(pred)
