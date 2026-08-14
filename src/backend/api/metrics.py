from fastapi import APIRouter
from src.backend.services.model_service import model_service

router = APIRouter()

@router.get("/metrics")
def get_metrics():
    return model_service.metrics
