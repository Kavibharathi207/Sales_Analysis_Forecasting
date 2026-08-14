"""
recommendations.py
==================
Generates actionable business recommendations per drug category based on:
  - Forecast trend direction and magnitude
  - Anomaly detection results
  - Model forecast error (MAE/MAPE from evaluation CSVs)
  - Sales volatility

Recommendation types:
  - RESTOCK_ALERT      : forecast shows significant demand increase
  - OVERSTOCK_RISK     : forecast shows significant demand decrease
  - DEMAND_SPIKE       : anomaly detected with high severity
  - STABLE_SUPPLY      : low volatility, forecast within normal range
  - HIGH_UNCERTAINTY   : model error is high, forecast less reliable
  - REVIEW_REQUIRED    : multiple conflicting signals

Usage
-----
    python src/features/recommendations.py
    python src/features/recommendations.py --categories M01AB,N02BE --model prophet
    python src/features/recommendations.py --model lightgbm --export
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = PROJECT_ROOT / "data" / "outputs" / "forecasts"
ANOMALY_DIR = PROJECT_ROOT / "data" / "outputs" / "anomalies"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "recommendations"
DATA_CSV = PROJECT_ROOT / "data" / "raw" / "pharma_sales_kaggle" / "salesdaily.csv"

ALL_CATEGORIES: List[str] = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
SUPPORTED_MODELS = ["prophet", "arima", "sarima", "lightgbm", "lstm"]

# Thresholds
TREND_UP_THRESHOLD = 10.0       # % forecast growth → restock alert
TREND_DOWN_THRESHOLD = -10.0    # % forecast decline → overstock risk
HIGH_MAPE_THRESHOLD = 30.0      # MAPE % above which forecast is unreliable
HIGH_VOLATILITY_THRESHOLD = 40.0  # CV% above which series is volatile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_forecast(category: str, model: str) -> Optional[pd.DataFrame]:
    """Load forecast CSV; return None if not found."""
    for m in [model] + [x for x in SUPPORTED_MODELS if x != model]:
        path = FORECAST_DIR / f"{category}_{m}_forecast.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["ds"])
            future = df[df["y"].isna()].copy() if "y" in df.columns else df.copy()
            if len(future) == 0:
                future = df.copy()
            return future[["ds", "yhat"]].reset_index(drop=True)
    return None


def load_evaluation(model: str) -> Optional[pd.DataFrame]:
    """Load model evaluation CSV (MAE, RMSE, MAPE per category)."""
    path = FORECAST_DIR / f"{model}_evaluation.csv"
    if path.exists():
        return pd.read_csv(path)
    # Try other models
    for m in SUPPORTED_MODELS:
        path = FORECAST_DIR / f"{m}_evaluation.csv"
        if path.exists():
            return pd.read_csv(path)
    return None


def load_anomaly_summary() -> Optional[pd.DataFrame]:
    """Load anomaly summary if it exists."""
    path = ANOMALY_DIR / "anomaly_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_historical_stats(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    """Compute mean and CV% for each category from raw data."""
    df = pd.read_csv(csv_path)
    rows = []
    for cat in ALL_CATEGORIES:
        if cat in df.columns:
            s = df[cat].dropna()
            mean = s.mean()
            std = s.std()
            rows.append({
                "category": cat,
                "hist_mean": round(mean, 4),
                "hist_std": round(std, 4),
                "cv_pct": round(100 * std / mean if mean != 0 else 0, 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------
def compute_forecast_trend(forecast_df: pd.DataFrame) -> float:
    """
    Compute % change from first to last forecast value.
    Positive = upward trend, negative = downward.
    """
    if len(forecast_df) < 2:
        return 0.0
    first = forecast_df["yhat"].iloc[:5].mean()   # smooth start
    last = forecast_df["yhat"].iloc[-5:].mean()   # smooth end
    if first == 0:
        return 0.0
    return round(100 * (last - first) / first, 2)


def get_mape(eval_df: Optional[pd.DataFrame], category: str) -> Optional[float]:
    """Extract MAPE for a category from evaluation DataFrame."""
    if eval_df is None:
        return None
    col_map = {"MAPE": "MAPE", "mape": "MAPE", "MAPE_%": "MAPE_%"}
    for col in ["MAPE", "mape", "MAPE_%"]:
        if col in eval_df.columns:
            cat_col = "category" if "category" in eval_df.columns else eval_df.columns[0]
            row = eval_df[eval_df[cat_col] == category]
            if len(row) > 0:
                val = row[col].iloc[0]
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
    return None


def get_anomaly_count(anomaly_summary: Optional[pd.DataFrame], category: str) -> int:
    """Get total anomaly count for a category."""
    if anomaly_summary is None:
        return 0
    row = anomaly_summary[anomaly_summary["category"] == category]
    if len(row) == 0:
        return 0
    return int(row["anomaly_count"].iloc[0])


def get_high_severity_anomalies(anomaly_summary: Optional[pd.DataFrame], category: str) -> int:
    if anomaly_summary is None:
        return 0
    row = anomaly_summary[anomaly_summary["category"] == category]
    if len(row) == 0 or "high_severity" not in anomaly_summary.columns:
        return 0
    return int(row["high_severity"].iloc[0])


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
def generate_recommendation(
    category: str,
    forecast_trend_pct: float,
    mape: Optional[float],
    anomaly_count: int,
    high_severity_anomalies: int,
    cv_pct: float,
    hist_mean: float,
) -> Dict:
    """
    Rule-based recommendation engine.

    Returns a dict with: category, signal, priority, recommendation, rationale
    """
    signals = []
    reasons = []

    # --- Trend signals ---
    if forecast_trend_pct >= TREND_UP_THRESHOLD:
        signals.append("RESTOCK_ALERT")
        reasons.append(f"Forecast shows +{forecast_trend_pct:.1f}% demand growth over horizon.")
    elif forecast_trend_pct <= TREND_DOWN_THRESHOLD:
        signals.append("OVERSTOCK_RISK")
        reasons.append(f"Forecast shows {forecast_trend_pct:.1f}% demand decline over horizon.")
    else:
        signals.append("STABLE_SUPPLY")
        reasons.append(f"Forecast trend is stable ({forecast_trend_pct:+.1f}%).")

    # --- Anomaly signals ---
    if high_severity_anomalies > 0:
        signals.append("DEMAND_SPIKE")
        reasons.append(f"{high_severity_anomalies} high-severity anomalies detected in historical data.")
    elif anomaly_count > 5:
        signals.append("DEMAND_SPIKE")
        reasons.append(f"{anomaly_count} anomalies detected — demand pattern is irregular.")

    # --- Model reliability ---
    if mape is not None and mape > HIGH_MAPE_THRESHOLD:
        signals.append("HIGH_UNCERTAINTY")
        reasons.append(f"Model MAPE is {mape:.1f}% — forecast reliability is low.")

    # --- Volatility ---
    if cv_pct > HIGH_VOLATILITY_THRESHOLD:
        reasons.append(f"High sales volatility (CV={cv_pct:.1f}%) — plan for demand variability.")

    # --- Priority ---
    if "RESTOCK_ALERT" in signals or "DEMAND_SPIKE" in signals:
        priority = "HIGH"
    elif "OVERSTOCK_RISK" in signals or "HIGH_UNCERTAINTY" in signals:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    # --- Conflicting signals ---
    if len(set(signals) - {"STABLE_SUPPLY"}) > 1:
        signals.append("REVIEW_REQUIRED")
        reasons.append("Multiple conflicting signals detected — manual review recommended.")

    # --- Primary recommendation text ---
    primary_signal = signals[0]
    rec_text = {
        "RESTOCK_ALERT": (
            f"Increase stock levels for {category}. "
            f"Demand is projected to grow by {forecast_trend_pct:.1f}% — "
            "ensure adequate inventory to avoid stockouts."
        ),
        "OVERSTOCK_RISK": (
            f"Consider reducing procurement for {category}. "
            f"Demand is projected to decline by {abs(forecast_trend_pct):.1f}% — "
            "excess inventory may lead to waste or write-offs."
        ),
        "DEMAND_SPIKE": (
            f"Investigate demand spikes for {category}. "
            "Historical anomalies suggest irregular demand events — "
            "review supply chain resilience and safety stock levels."
        ),
        "STABLE_SUPPLY": (
            f"Maintain current procurement strategy for {category}. "
            "Demand is stable and forecast confidence is acceptable."
        ),
        "HIGH_UNCERTAINTY": (
            f"Use caution when planning for {category}. "
            "Forecast model error is high — supplement with domain expertise."
        ),
        "REVIEW_REQUIRED": (
            f"Manual review required for {category}. "
            "Multiple signals detected — consult supply chain team."
        ),
    }.get(primary_signal, "No specific recommendation.")

    return {
        "category": category,
        "signal": " | ".join(signals),
        "priority": priority,
        "forecast_trend_pct": forecast_trend_pct,
        "mape": round(mape, 2) if mape is not None else "N/A",
        "anomaly_count": anomaly_count,
        "cv_pct": cv_pct,
        "hist_mean_sales": round(hist_mean, 2),
        "recommendation": rec_text,
        "rationale": " | ".join(reasons),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    categories: Optional[List[str]] = None,
    model: str = "prophet",
    export: bool = False,
) -> pd.DataFrame:
    if categories is None:
        categories = ALL_CATEGORIES

    eval_df = load_evaluation(model)
    anomaly_summary = load_anomaly_summary()
    hist_stats = load_historical_stats()
    hist_map = hist_stats.set_index("category").to_dict("index")

    if anomaly_summary is None:
        logger.warning(
            "Anomaly summary not found. Run anomaly_detection.py first for richer recommendations."
        )

    all_recs = []
    for cat in categories:
        forecast_df = load_forecast(cat, model)
        if forecast_df is None:
            logger.warning("[%s] No forecast found, skipping.", cat)
            continue

        trend = compute_forecast_trend(forecast_df)
        mape = get_mape(eval_df, cat)
        anomaly_count = get_anomaly_count(anomaly_summary, cat)
        high_sev = get_high_severity_anomalies(anomaly_summary, cat)
        stats = hist_map.get(cat, {"cv_pct": 0.0, "hist_mean": 0.0})

        rec = generate_recommendation(
            category=cat,
            forecast_trend_pct=trend,
            mape=mape,
            anomaly_count=anomaly_count,
            high_severity_anomalies=high_sev,
            cv_pct=stats.get("cv_pct", 0.0),
            hist_mean=stats.get("hist_mean", 0.0),
        )
        all_recs.append(rec)
        logger.info("[%s] Priority=%s | Signal=%s", cat, rec["priority"], rec["signal"])

    if not all_recs:
        logger.warning("No recommendations generated.")
        return pd.DataFrame()

    rec_df = pd.DataFrame(all_recs)

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    rec_df["_sort"] = rec_df["priority"].map(priority_order)
    rec_df.sort_values("_sort", inplace=True)
    rec_df.drop(columns=["_sort"], inplace=True)

    if export:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "recommendations.csv"
        rec_df.to_csv(out_path, index=False)
        logger.info("Recommendations saved -> %s", out_path)

    print("\n" + "=" * 70)
    print("  SALES RECOMMENDATIONS")
    print("=" * 70)
    for _, row in rec_df.iterrows():
        print(f"\n[{row['priority']}] {row['category']}  |  {row['signal']}")
        print(f"  → {row['recommendation']}")
        print(f"  Rationale: {row['rationale']}")
    print("\n" + "=" * 70)

    return rec_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate sales recommendations from forecasts.")
    p.add_argument("--categories", type=str, default=None, help="Comma-separated ATC codes")
    p.add_argument("--model", type=str, default="prophet", choices=SUPPORTED_MODELS)
    p.add_argument("--export", action="store_true", help="Save recommendations to CSV")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    run_pipeline(categories=cats, model=args.model, export=args.export)


if __name__ == "__main__":
    main()
