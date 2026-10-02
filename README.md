# Freight Rate Prediction

Predicts `posted_rate` for truckload freight. The model is trained on `data/train_test.csv` (Jan–Oct 2025) and predicts the 12,000 loads in `data/validation.csv` (Nov–Dec 2025) plus the 31 rows of the fixed December chart.

**Result:** a LightGBM model on log(rate). On a time-based rolling backtest with a 2-month horizon it reaches a mean **MAPE of 2.21% (MAE $51)**. The distance-band rate-per-mile baseline scores 3.53%, and the same model on raw features scores 3.31%.

## Run

```bash
python -m pip install -r requirements.txt
python train.py      # backtest + fit final model  -> models/, reports/metrics.json, reports/backtest.csv
python predict.py    # -> validation_predictions.csv, fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python eda.py        # report figures -> reports/figures/
python build_report.py  # -> reports/Freight_Rate_Report.docx
```

Tested on Python 3.14, LightGBM 4.7, pandas 2.3, scikit-learn 1.8. The full run takes about 1 minute on a laptop.

## Layout

```
src/freight_rate/
  config.py     paths, constants, model hyperparameters, backtest folds
  data.py       loading + data-quality fixes (weights, missing market_index, corrupted labels)
  features.py   feature engineering (market signal decomposition)
  model.py      FreightRateModel (LightGBM), DistanceBandBaseline, metrics
train.py        rolling-origin backtest, random-split comparison, final fit
predict.py      validation predictions + December chart inputs
eda.py          figures for the report
build_report.py DOCX report
score.py        provided scorer (unchanged)
```

## Key findings

| Issue | Rows | Handling |
|---|---|---|
| Corrupted `posted_rate` (off by 0.2–0.45× or 2–5×) | 677 (1.4%) | flagged against an out-of-fold model (>1.5× away), dropped from training and evaluation |
| Negative `weight` | 292 train / 145 val | sign error → absolute value (these loads price the same as positive-weight loads) |
| Missing `weight` | 300 / 165 | median weight for the equipment type (from training) |
| Missing `market_index` | 374 / 249 | that day's median (`market_index` is 98% explained by date) |
| Cities in validation that never appear in training | 8 | no city-ID features; the model uses coordinates and distance |
| Coordinates clipped (lat 25.5, lon −69.5) | 3 cities | consistent per city, so used as relative geography |

* **`market_index` is a trap feature.** Its daily level swings about 40% (a 7-day cycle plus a spring hump), while rates move about 10%. Fed in raw, it makes the model worse on future months. Only each load's deviation from the daily median is kept.
* **`quote_signal`** is split the same way into a daily level and a load-level deviation. The deviation adds most of the gain (backtest MAPE 2.6% → 2.2%).
* **Distance** explains about 83% of the gain: rate per mile falls from about $2.8 under 200 miles to about $1.9 over 2,000 miles. Reefer is about 13% higher than Dry Van per mile, and Flatbed about 8% higher.

## Validation

The data is a time series and the real task is a 2-month forecast, so validation is a **rolling-origin backtest**:

| Fold | Train | Test |
|---|---|---|
| 1 | Jan–Jun | Jul–Aug |
| 2 | Jan–Jul | Aug–Sep |
| 3 | Jan–Aug | Sep–Oct |

A random 80/20 split gives 2.11% MAPE, which is optimistic because it sees every month's market regime during training. The final model is refit on all clean Jan–Oct data.

## December chart

`december_chart_inputs.csv` only provides lane, distance, equipment, weight and date. The model inputs it lacks are filled with their expected value for that date. Coordinates come from a city lookup. `market_index` and `quote_signal` are set to the daily median across all validation loads on that day, so their load-level deviations are 0. The chart therefore shows the model's market-level view for one fixed load.
