"""
recommendation_service.py
=========================
Rule-based recommendation engine.

Inputs  : forecast series + optional anomaly summary + optional model MAPE
Outputs : prioritised list of actionable recommendations

Rules (initial team thresholds — adjust as needed):
  Forecast trend > +10%   → RESTOCK_ALERT      (HIGH)
  Forecast trend < -10%   → OVERSTOCK_RISK     (MEDIUM)
  High-severity anomalies → DEMAND_SPIKE       (HIGH)
  Model MAPE > 30%        → HIGH_UNCERTAINTY   (MEDIUM)
  Otherwise               → STABLE_SUPPLY      (LOW)
"""

from typing import List
from backend.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    Recommendation,
)

TREND_UP = 10.0
TREND_DOWN = -10.0
HIGH_MAPE = 30.0


def _compute_trend(forecast) -> float:
    """% change from first 5-day avg to last 5-day avg."""
    if len(forecast) < 2:
        return 0.0
    first = sum(p.predicted_sales for p in forecast[:5]) / min(5, len(forecast))
    last = sum(p.predicted_sales for p in forecast[-5:]) / min(5, len(forecast))
    if first == 0:
        return 0.0
    return round((last - first) / first * 100, 2)


def generate_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    trend = _compute_trend(request.forecast)
    recs: List[Recommendation] = []

    # --- Trend signals ---
    if trend >= TREND_UP:
        recs.append(Recommendation(
            signal="RESTOCK_ALERT",
            priority="HIGH",
            recommendation=(
                f"Increase stock levels for {request.product}. "
                f"Demand is projected to grow by {trend:.1f}% — "
                "ensure adequate inventory to avoid stockouts."
            ),
            rationale=f"Forecast trend is +{trend:.1f}% over the horizon.",
        ))
    elif trend <= TREND_DOWN:
        recs.append(Recommendation(
            signal="OVERSTOCK_RISK",
            priority="MEDIUM",
            recommendation=(
                f"Consider reducing procurement for {request.product}. "
                f"Demand is projected to decline by {abs(trend):.1f}% — "
                "excess inventory may lead to waste."
            ),
            rationale=f"Forecast trend is {trend:.1f}% over the horizon.",
        ))
    else:
        recs.append(Recommendation(
            signal="STABLE_SUPPLY",
            priority="LOW",
            recommendation=(
                f"Maintain current procurement strategy for {request.product}. "
                "Demand is stable within normal range."
            ),
            rationale=f"Forecast trend is {trend:+.1f}% — within ±10% threshold.",
        ))

    # --- Anomaly signals ---
    if request.anomaly_summary:
        if request.anomaly_summary.high_severity_count > 0:
            recs.append(Recommendation(
                signal="DEMAND_SPIKE",
                priority="HIGH",
                recommendation=(
                    f"Investigate demand spikes for {request.product}. "
                    f"{request.anomaly_summary.high_severity_count} high-severity anomalies detected — "
                    "review supply chain resilience and safety stock."
                ),
                rationale=(
                    f"{request.anomaly_summary.high_severity_count} high-severity anomalies "
                    f"out of {request.anomaly_summary.anomaly_count} total anomalies."
                ),
            ))
        elif request.anomaly_summary.anomaly_count > 3:
            recs.append(Recommendation(
                signal="DEMAND_SPIKE",
                priority="MEDIUM",
                recommendation=(
                    f"Monitor demand irregularities for {request.product}. "
                    f"{request.anomaly_summary.anomaly_count} anomalies detected in the forecast period."
                ),
                rationale=f"{request.anomaly_summary.anomaly_count} anomalies detected.",
            ))

    # --- Model reliability ---
    if request.model_mape is not None and request.model_mape > HIGH_MAPE:
        recs.append(Recommendation(
            signal="HIGH_UNCERTAINTY",
            priority="MEDIUM",
            recommendation=(
                f"Use caution when planning for {request.product}. "
                f"Forecast model MAPE is {request.model_mape:.1f}% — "
                "supplement with domain expertise before making procurement decisions."
            ),
            rationale=f"Model MAPE {request.model_mape:.1f}% exceeds the {HIGH_MAPE}% reliability threshold.",
        ))

    # Sort: HIGH first, then MEDIUM, then LOW
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recs.sort(key=lambda r: priority_order.get(r.priority, 3))

    return RecommendationResponse(
        product=request.product,
        forecast_trend_pct=trend,
        recommendations=recs,
    )
