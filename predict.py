"""Predict the 12,000 validation loads and the 31 fixed December chart rows.

    python predict.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from freight_rate import config  # noqa: E402
from freight_rate.data import clean_weight, daily_market_stats, load  # noqa: E402


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for df in frames:
        for side in ["pickup", "delivery"]:
            parts.append(df[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def predict_validation(bundle, validation: pd.DataFrame) -> pd.DataFrame:
    features = clean_weight(validation, bundle["weight_medians"])
    preds = pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": bundle["model"].predict(features)})

    template = pd.read_csv(config.TEMPLATE_PATH)
    out = template[["load_id"]].merge(preds, on="load_id", how="left", validate="one_to_one")
    assert len(out) == 12_000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out["predicted_rate"] = out["predicted_rate"].round(2)
    return out


def predict_december(bundle, validation: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """The chart file only has lane, distance, equipment, weight and date.
    Missing model inputs are filled with their expected value for that date:
    coordinates from the city lookup, and market_index / quote_signal with the
    daily market level observed across all validation loads on the same day
    (so their load-level deviations are 0)."""
    december = pd.read_csv(config.DECEMBER_PATH, parse_dates=["date"])
    coords = city_coordinates(train, validation)
    market = daily_market_stats(validation)

    X = december.copy()
    for side in ["pickup", "delivery"]:
        X[f"{side}_lat"] = X[side].map(coords["lat"])
        X[f"{side}_lon"] = X[side].map(coords["lon"])
    X = X.merge(market, on="date", how="left")
    X["market_index"] = X["market_index_day"]
    X["quote_signal"] = X["quote_signal_day"]
    assert X.notna().drop(columns="predicted_rate").all().all()

    december["predicted_rate"] = bundle["model"].predict(X, market_stats=market).round(2)
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    return december


def main() -> None:
    # joblib unpickles code: only load the artifact this repo's train.py produced.
    bundle = joblib.load(config.MODEL_PATH)
    validation = load(config.VALIDATION_PATH)
    train = load(config.TRAIN_PATH)

    preds = predict_validation(bundle, validation)
    preds.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(preds):,} predictions to {config.PREDICTIONS_PATH}")
    print(preds["predicted_rate"].describe().round(2).to_string())

    december = predict_december(bundle, validation, train)
    december.to_csv(config.DECEMBER_PATH, index=False)
    print(f"\nFilled December chart inputs: {config.DECEMBER_PATH}")
    print(december[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
