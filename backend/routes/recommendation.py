from fastapi import APIRouter
from backend.schemas.recommendation import RecommendationRequest, RecommendationResponse
from backend.services.recommendation_service import generate_recommendations

router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])


@router.post("", response_model=RecommendationResponse)
def recommendations(request: RecommendationRequest):
    """
    Generate actionable recommendations based on:
    - Forecast trend direction
    - Anomaly summary (optional)
    - Model MAPE / reliability (optional)
    """
    return generate_recommendations(request)
