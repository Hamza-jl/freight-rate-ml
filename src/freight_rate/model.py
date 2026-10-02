"""Model, baseline and metrics."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .features import build_features, inverse_target, log_target
from .data import daily_market_stats


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(100 * np.mean(np.abs(err) / y_true)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "n": int(len(y_true)),
    }


class FreightRateModel:
    """LightGBM on log(posted_rate). Log target makes errors relative, which suits
    rates spanning $60-$8k, and lets trees model a rate-per-mile structure."""

    def __init__(self, params: dict | None = None):
        self.params = params or config.LGBM_PARAMS
        self.model: lgb.LGBMRegressor | None = None

    def fit(self, train: pd.DataFrame) -> "FreightRateModel":
        X = build_features(train, daily_market_stats(train))
        self.model = lgb.LGBMRegressor(**self.params).fit(X, log_target(train[config.TARGET]))
        return self

    def predict(self, df: pd.DataFrame, market_stats: pd.DataFrame | None = None) -> np.ndarray:
        stats = daily_market_stats(df) if market_stats is None else market_stats
        return inverse_target(self.model.predict(build_features(df, stats)))

    def feature_importance(self) -> pd.Series:
        gain = self.model.booster_.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=self.model.feature_name_).sort_values(ascending=False)


class DistanceBandBaseline:
    """Median rate-per-mile by equipment x 100-mile distance band, times distance."""

    bins = np.r_[0, np.arange(100, 3600, 100), np.inf]

    def fit(self, train: pd.DataFrame) -> "DistanceBandBaseline":
        rpm = train[config.TARGET] / train["distance"]
        band = pd.cut(train["distance"], self.bins)
        self.table = rpm.groupby([train["equipment"], band], observed=True).median()
        self.fallback = rpm.median()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        band = pd.cut(df["distance"], self.bins)
        keys = pd.MultiIndex.from_arrays([df["equipment"], band])
        rpm = self.table.reindex(keys).fillna(self.fallback).to_numpy()
        return df["distance"].to_numpy() * rpm
