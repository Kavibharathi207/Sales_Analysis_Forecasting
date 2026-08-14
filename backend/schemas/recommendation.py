from pydantic import BaseModel
from typing import List, Optional


class ForecastPoint(BaseModel):
    date: str
    predicted_sales: float


class AnomalySummary(BaseModel):
    anomaly_count: int
    high_severity_count: int


class RecommendationRequest(BaseModel):
    product: str
    forecast: List[ForecastPoint]
    anomaly_summary: Optional[AnomalySummary] = None
    model_mape: Optional[float] = None          # from evaluation CSV


class Recommendation(BaseModel):
    signal: str          # e.g. "RESTOCK_ALERT"
    priority: str        # "HIGH" | "MEDIUM" | "LOW"
    recommendation: str  # human-readable action
    rationale: str       # why this was triggered


class RecommendationResponse(BaseModel):
    product: str
    forecast_trend_pct: float
    recommendations: List[Recommendation]
