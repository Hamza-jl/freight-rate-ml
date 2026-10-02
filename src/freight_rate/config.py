"""Paths and constants shared by the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
MODEL_PATH = MODEL_DIR / "lgbm_freight_rate.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

TARGET = "posted_rate"
SEED = 42

# A label is treated as corrupted when it is more than 1.5x away from an
# out-of-fold prediction. Clean labels sit within ~+/-7% (99th pct), corrupted
# ones are off by 2-5x, so the threshold is not sensitive.
CORRUPTION_RATIO = 1.5

# Rolling-origin backtest: train on months <= m, test on the next two months.
# Two months mirrors the real task (train Jan-Oct, predict Nov-Dec).
BACKTEST_FOLDS = [(6, 8), (7, 9), (8, 10)]

LGBM_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
)
