"""Loading and cleaning.

Data-quality issues found during EDA and how they are handled:

* weight < 0 (~0.6% of rows): sign error -> absolute value. Negative-weight
  loads price the same as positive ones, so the magnitude is real.
* weight missing (~0.6%): imputed with the training median for the equipment type.
* market_index missing (~0.8%): ~98% of its variance is a daily, market-wide
  value, so it is imputed with that day's median across all loads.
* posted_rate corrupted (~1.4%): labels off by 0.2-0.45x or 2-5x versus an
  out-of-fold model. Dropped from training and from evaluation.
* Coordinates are clipped at lat 25.5 / lon -69.5 (Laredo, Boston, Providence)
  and do not match real-world locations; they are still consistent per city,
  so they are used as-is as relative geography.
* 8 cities in validation never appear in training -> no city-ID features;
  the model only sees coordinates and distance, which generalise to new cities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def fit_weight_medians(train: pd.DataFrame) -> dict[str, float]:
    return train["weight"].abs().groupby(train["equipment"]).median().to_dict()


def clean_weight(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = df["weight"].abs()
    df["weight"] = df["weight"].fillna(df["equipment"].map(medians))
    return df


def daily_market_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day median market_index and quote_signal across all loads that day."""
    return (
        df.groupby("date")
        .agg(market_index_day=("market_index", "median"), quote_signal_day=("quote_signal", "median"))
        .reset_index()
    )


def flag_corrupted_labels(df: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    """True where posted_rate is >1.5x away from an out-of-fold robust model."""
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    X = pd.DataFrame(
        {
            "log_distance": np.log(df["distance"]),
            "weight": df["weight"],
            "quote_signal": df["quote_signal"],
            "equipment": df["equipment"].astype("category"),
        }
    )
    y = np.log(df[config.TARGET])
    oof = np.zeros(len(df))
    for fit_idx, oof_idx in KFold(n_splits, shuffle=True, random_state=config.SEED).split(X):
        model = lgb.LGBMRegressor(n_estimators=400, objective="huber", alpha=0.1, verbose=-1)
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        oof[oof_idx] = model.predict(X.iloc[oof_idx])
    return pd.Series(np.abs(y - oof) > np.log(config.CORRUPTION_RATIO), index=df.index)


def load_clean_training() -> tuple[pd.DataFrame, dict[str, float]]:
    raw = load(config.TRAIN_PATH)
    medians = fit_weight_medians(raw)
    df = clean_weight(raw, medians)
    df["is_corrupted"] = flag_corrupted_labels(df)
    return df, medians
