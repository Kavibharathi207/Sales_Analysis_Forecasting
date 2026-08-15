"""
recommendation_service.py
=========================
Loads teammate's forecast, actual sales, and model evaluation
from disk — then generates rule-based recommendations.

Pipeline:
    1. Load forecast from disk  → compute trend
    2. Load actuals from disk   → run anomaly detection internally
    3. Load MAPE from eval CSV  → assess model reliability
    4. Apply rules              → return prioritised recommendations

Rules:
    Forecast trend > +10%          → RESTOCK_ALERT      (HIGH)
    Forecast trend < -10%          → OVERSTOCK_RISK     (MEDIUM)
    High-severity anomalies exist  → DEMAND_SPIKE       (HIGH)
    Model MAPE > 30%               → HIGH_UNCERTAINTY   (MEDIUM)
    Otherwise                      → STABLE_SUPPLY      (LOW)
"""

from backend.data_loader import load_forecast, load_test_period, load_actuals, load_mape
from backend.schemas.recommendation import (
    RecommendationRequest, RecommendationResponse, Recommendation,
)

TREND_UP   = 10.0
TREND_DOWN = -10.0
HIGH_MAPE  = 30.0
ANOMALY_HIGH_THRESHOLD = 50.0   # deviation% above which severity = high


def _compute_trend(forecast_df) -> float:
    """% change from first-5-day avg to last-5-day avg of forecast."""
    if len(forecast_df) < 2:
        return 0.0
    first = forecast_df["predicted_sales"].iloc[:5].mean()
    last  = forecast_df["predicted_sales"].iloc[-5:].mean()
    if first == 0:
        return 0.0
    return round((last - first) / first * 100, 2)


def _count_anomalies(forecast_df, actuals_df):
    """Quick internal anomaly count — no need to call the full anomaly service."""
    merged = forecast_df.merge(actuals_df, on="date", how="inner")
    total = high = 0
    for _, row in merged.iterrows():
        f = float(row["predicted_sales"])
        a = float(row["actual_sales"])
        if f == 0:
            dev = 0.0 if a == 0 else 100.0
        else:
            dev = abs(a - f) / f * 100
        if dev >= 25.0:
            total += 1
        if dev > ANOMALY_HIGH_THRESHOLD:
            high += 1
    return total, high


def generate_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    # Load all data from disk
    forecast_df = load_forecast(request.category, request.model)   # future rows → trend
    test_df     = load_test_period(request.category, request.model) # test rows → anomaly count
    actuals_df  = load_actuals(request.category)
    mape        = load_mape(request.category, request.model)

    trend = _compute_trend(forecast_df)
    anomaly_count, high_severity_count = _count_anomalies(test_df, actuals_df)

    recs = []

    # --- Trend signal ---
    if trend >= TREND_UP:
        recs.append(Recommendation(
            signal="RESTOCK_ALERT",
            priority="HIGH",
            recommendation=(
                f"Increase stock levels for {request.category}. "
                f"Demand is projected to grow by {trend:.1f}% — "
                "ensure adequate inventory to avoid stockouts."
            ),
            rationale=f"Forecast trend is +{trend:.1f}% over the forecast horizon.",
        ))
    elif trend <= TREND_DOWN:
        recs.append(Recommendation(
            signal="OVERSTOCK_RISK",
            priority="MEDIUM",
            recommendation=(
                f"Consider reducing procurement for {request.category}. "
                f"Demand is projected to decline by {abs(trend):.1f}% — "
                "excess inventory may lead to waste or write-offs."
            ),
            rationale=f"Forecast trend is {trend:.1f}% over the forecast horizon.",
        ))
    else:
        recs.append(Recommendation(
            signal="STABLE_SUPPLY",
            priority="LOW",
            recommendation=(
                f"Maintain current procurement strategy for {request.category}. "
                "Demand is stable within the normal range."
            ),
            rationale=f"Forecast trend is {trend:+.1f}% — within the ±10% stable threshold.",
        ))

    # --- Anomaly signal ---
    if high_severity_count > 0:
        recs.append(Recommendation(
            signal="DEMAND_SPIKE",
            priority="HIGH",
            recommendation=(
                f"Investigate demand spikes for {request.category}. "
                f"{high_severity_count} high-severity anomaly days detected — "
                "review safety stock and supply chain resilience."
            ),
            rationale=(
                f"{high_severity_count} high-severity anomalies "
                f"(deviation > 50%) out of {anomaly_count} total anomaly days."
            ),
        ))
    elif anomaly_count > 3:
        recs.append(Recommendation(
            signal="DEMAND_SPIKE",
            priority="MEDIUM",
            recommendation=(
                f"Monitor demand irregularities for {request.category}. "
                f"{anomaly_count} anomaly days detected in the overlapping period."
            ),
            rationale=f"{anomaly_count} days with deviation > 25% detected.",
        ))

    # --- Model reliability signal ---
    if mape is not None and mape > HIGH_MAPE:
        recs.append(Recommendation(
            signal="HIGH_UNCERTAINTY",
            priority="MEDIUM",
            recommendation=(
                f"Use caution when planning for {request.category}. "
                f"The {request.model} model MAPE is {mape:.1f}% — "
                "supplement forecasts with domain expertise before procurement decisions."
            ),
            rationale=f"Model MAPE {mape:.1f}% exceeds the {HIGH_MAPE}% reliability threshold.",
        ))

    # Sort HIGH → MEDIUM → LOW
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recs.sort(key=lambda r: order.get(r.priority, 3))

    return RecommendationResponse(
        category=request.category,
        model=request.model,
        forecast_trend_pct=trend,
        model_mape=mape,
        anomaly_count=anomaly_count,
        recommendations=recs,
    )
