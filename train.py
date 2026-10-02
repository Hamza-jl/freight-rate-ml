"""Validate the approach with a time-based backtest, then fit and save the final model.

    python train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from freight_rate import config  # noqa: E402
from freight_rate.data import load_clean_training  # noqa: E402
from freight_rate.model import DistanceBandBaseline, FreightRateModel, metrics  # noqa: E402


class NaiveLGBM:
    """Same model fed the raw columns (incl. raw market_index) - the ablation that
    shows why the market features are decomposed."""

    cols = ["distance", "weight", "market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

    def _X(self, df):
        X = df[self.cols].copy()
        X["equipment"] = pd.Categorical(df["equipment"], categories=["Dry Van", "Flatbed", "Reefer"])
        return X

    def fit(self, df):
        self.m = lgb.LGBMRegressor(**config.LGBM_PARAMS).fit(self._X(df), np.log(df[config.TARGET]))
        return self

    def predict(self, df):
        return np.exp(self.m.predict(self._X(df)))


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[~df["is_corrupted"]]
    rows = []
    for last_train_month, last_test_month in config.BACKTEST_FOLDS:
        month = clean["date"].dt.month
        train = clean[month <= last_train_month]
        test = clean[(month > last_train_month) & (month <= last_test_month)]
        fold = f"train Jan-{train['date'].max():%b} -> test {test['date'].min():%b}-{test['date'].max():%b}"
        for name, model in [
            ("Baseline: rate/mile by equipment x distance band", DistanceBandBaseline()),
            ("LightGBM, raw features", NaiveLGBM()),
            ("LightGBM, final features", FreightRateModel()),
        ]:
            pred = model.fit(train).predict(test)
            rows.append({"fold": fold, "model": name, **metrics(test[config.TARGET], pred)})
        print(f"  done: {fold}")
    return pd.DataFrame(rows)


def random_split_check(df: pd.DataFrame) -> dict:
    """Random 80/20 split, to show how optimistic it is versus the time split."""
    clean = df[~df["is_corrupted"]]
    test = clean.sample(frac=0.2, random_state=config.SEED)
    train = clean.drop(test.index)
    return metrics(test[config.TARGET], FreightRateModel().fit(train).predict(test))


def main() -> None:
    df, weight_medians = load_clean_training()
    n_bad = int(df["is_corrupted"].sum())
    print(f"Loaded {len(df):,} training rows; flagged {n_bad} ({n_bad / len(df):.1%}) corrupted labels")

    print("Rolling-origin backtest (2-month horizon):")
    bt = backtest(df)
    summary = bt.groupby("model", sort=False)[["MAE", "MAPE", "RMSE"]].mean()
    print(bt.round(2).to_string(index=False))
    print("\nMean over folds:\n" + summary.round(2).to_string())

    # Headline holdout = last fold (Jan-Aug -> Sep-Oct). Also score it against the
    # raw labels (corrupted rows included) for transparency.
    clean = df[~df["is_corrupted"]]
    hold_train = clean[clean["date"].dt.month <= 8]
    hold_test_raw = df[df["date"].dt.month > 8]
    holdout_model = FreightRateModel().fit(hold_train)
    raw_metrics = metrics(hold_test_raw[config.TARGET], holdout_model.predict(hold_test_raw))

    random_metrics = random_split_check(df)
    print(f"\nRandom 80/20 split MAPE: {random_metrics['MAPE']:.2f}% (optimistic; leaks future market regime)")

    final = FreightRateModel().fit(clean)
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "weight_medians": weight_medians}, config.MODEL_PATH)
    importance = final.feature_importance()
    print("\nFeature importance (gain share):\n" + importance.round(3).to_string())

    config.REPORT_DIR.mkdir(exist_ok=True)
    bt.to_csv(config.REPORT_DIR / "backtest.csv", index=False)
    config.METRICS_PATH.write_text(
        json.dumps(
            {
                "n_train_rows": len(df),
                "n_corrupted_labels": n_bad,
                "backtest_mean": summary.round(3).to_dict(orient="index"),
                "holdout_sep_oct_raw_labels": raw_metrics,
                "random_split": random_metrics,
                "feature_importance": importance.round(4).to_dict(),
            },
            indent=2,
        )
    )
    print(f"\nSaved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
