"""
build_metrics_json.py
---------------------
Merges all per-model evaluation CSVs produced by the ML team into a single
metrics.json consumed by the backend /metrics and /compare-models endpoints.

ML team produces:
    data/outputs/forecasts/prophet_evaluation.csv
    data/outputs/forecasts/arima_evaluation.csv
    data/outputs/forecasts/sarima_evaluation.csv
    data/outputs/forecasts/lightgbm_evaluation.csv
    data/outputs/forecasts/lstm_evaluation.csv

Output:
    data/outputs/metrics.json

Run:
    python src/backend/build_metrics_json.py
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = PROJECT_ROOT / "data" / "outputs" / "forecasts"
OUT_FILE     = PROJECT_ROOT / "data" / "outputs" / "metrics.json"

MODEL_CSV_MAP = {
    "prophet":   FORECAST_DIR / "prophet_evaluation.csv",
    "arima":     FORECAST_DIR / "arima_evaluation.csv",
    "sarima":    FORECAST_DIR / "sarima_evaluation.csv",
    "lightgbm":  FORECAST_DIR / "lightgbm_evaluation.csv",
    "lstm":      FORECAST_DIR / "lstm_evaluation.csv",
}


def load_eval_csv(model_type: str, path: Path) -> dict:
    """Read one evaluation CSV and return {category: {MAE, RMSE, MAPE}}."""
    if not path.exists():
        logger.warning("Missing: %s — skipping %s", path.name, model_type)
        return {}

    df = pd.read_csv(path)

    # Normalise column names to uppercase
    df.columns = [c.strip().upper() for c in df.columns]

    if "CATEGORY" not in df.columns:
        logger.warning("%s has no 'category' column — skipping", path.name)
        return {}

    result = {}
    for _, row in df.iterrows():
        cat = str(row["CATEGORY"]).strip()
        result[cat] = {
            "MAE":  _safe_float(row.get("MAE")),
            "RMSE": _safe_float(row.get("RMSE")),
            "MAPE": _safe_float(row.get("MAPE")),
        }
    logger.info("Loaded %d categories from %s", len(result), path.name)
    return result


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def pick_best_model(category_metrics: dict) -> str:
    """Return the model name with the lowest MAE for a given category."""
    best_model, best_mae = None, float("inf")
    for model_type, metrics in category_metrics.items():
        mae = metrics.get("MAE")
        if mae is not None and mae < best_mae:
            best_mae = mae
            best_model = model_type
    return best_model or "unknown"


def build_metrics_json():
    # Load all available evaluation CSVs
    all_data: dict[str, dict] = {}  # { model_type: { category: metrics } }
    for model_type, csv_path in MODEL_CSV_MAP.items():
        all_data[model_type] = load_eval_csv(model_type, csv_path)

    # Collect all categories across all models
    all_categories = set()
    for cat_dict in all_data.values():
        all_categories.update(cat_dict.keys())

    if not all_categories:
        logger.error("No evaluation data found. Run the ML training scripts first.")
        return

    # Build output structure
    output = {}
    for cat in sorted(all_categories):
        cat_entry = {}
        for model_type, cat_dict in all_data.items():
            if cat in cat_dict:
                cat_entry[model_type] = cat_dict[cat]

        if cat_entry:
            cat_entry["best_model"] = pick_best_model(cat_entry)
            output[cat] = cat_entry

    # Write
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("metrics.json written -> %s  (%d categories)", OUT_FILE, len(output))

    # Print summary
    print("\n" + "=" * 60)
    print("  METRICS SUMMARY")
    print("=" * 60)
    for cat, data in output.items():
        best = data.get("best_model", "?")
        models_available = [m for m in data if m != "best_model"]
        print(f"  {cat:<10}  best={best:<12}  models={models_available}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    build_metrics_json()
