from fastapi import APIRouter
from backend.schemas.anomaly import AnomalyDetectRequest, AnomalyDetectResponse
from backend.services.anomaly_service import detect_anomalies

router = APIRouter(prefix="/api/anomaly", tags=["Anomaly Detection"])


@router.post("/detect", response_model=AnomalyDetectResponse)
def detect(request: AnomalyDetectRequest):
    """
    Detect anomalies by comparing forecast vs actual sales.

    Thresholds:
    - deviation < 10%  → normal
    - 10–25%           → moderate
    - > 25%            → anomaly (high if > 50%)
    """
    return detect_anomalies(request)
