from pydantic import BaseModel
from typing import List, Optional


class ForecastPoint(BaseModel):
    date: str
    predicted_sales: float


class ActualPoint(BaseModel):
    date: str
    actual_sales: float


class AnomalyDetectRequest(BaseModel):
    product: str
    forecast: List[ForecastPoint]
    actuals: List[ActualPoint]


class AnomalyResult(BaseModel):
    date: str
    actual_sales: float
    forecast_sales: float
    deviation_percent: float
    status: str        # "normal" | "moderate" | "anomaly"
    severity: str      # "low" | "medium" | "high"


class AnomalyDetectResponse(BaseModel):
    product: str
    total_days: int
    anomaly_count: int
    results: List[AnomalyResult]
