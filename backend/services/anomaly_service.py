"""
anomaly_service.py
==================
Business logic for anomaly detection.

Algorithm
---------
deviation_percent = abs(actual - forecast) / forecast * 100

Thresholds (agreed team rules — adjust as needed):
  < 10%   → normal
  10–25%  → moderate  (low severity)
  > 25%   → anomaly   (medium if 25–50%, high if > 50%)
"""

from typing import List
from backend.schemas.anomaly import (
    AnomalyDetectRequest,
    AnomalyDetectResponse,
    AnomalyResult,
)


MODERATE_THRESHOLD = 10.0
ANOMALY_THRESHOLD = 25.0
HIGH_THRESHOLD = 50.0


def detect_anomalies(request: AnomalyDetectRequest) -> AnomalyDetectResponse:
    # Build lookup: date → actual_sales
    actual_map = {a.date: a.actual_sales for a in request.actuals}

    results: List[AnomalyResult] = []

    for fp in request.forecast:
        actual = actual_map.get(fp.date)
        if actual is None:
            continue  # no actual for this date — skip

        forecast = fp.predicted_sales

        # Avoid division by zero
        if forecast == 0:
            deviation = 0.0 if actual == 0 else 100.0
        else:
            deviation = abs(actual - forecast) / forecast * 100

        deviation = round(deviation, 2)

        if deviation < MODERATE_THRESHOLD:
            status, severity = "normal", "low"
        elif deviation < ANOMALY_THRESHOLD:
            status, severity = "moderate", "low"
        elif deviation < HIGH_THRESHOLD:
            status, severity = "anomaly", "medium"
        else:
            status, severity = "anomaly", "high"

        results.append(AnomalyResult(
            date=fp.date,
            actual_sales=actual,
            forecast_sales=round(forecast, 4),
            deviation_percent=deviation,
            status=status,
            severity=severity,
        ))

    anomaly_count = sum(1 for r in results if r.status == "anomaly")

    return AnomalyDetectResponse(
        product=request.product,
        total_days=len(results),
        anomaly_count=anomaly_count,
        results=results,
    )
