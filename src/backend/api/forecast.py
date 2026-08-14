from fastapi import APIRouter, HTTPException
from src.backend.schemas.forecast import ForecastRequest, CompareRequest
from src.backend.services.model_service import model_service
from src.backend.services.forecast_service import generate_forecast

router = APIRouter()


@router.post("/forecast")
def forecast(req: ForecastRequest):
    model = model_service.get_model(req.model, req.category)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {req.model} model available for category '{req.category}'"
        )
    scaler = model_service.get_scaler(req.category) if req.model == "lstm" else None
    try:
        predictions = generate_forecast(model, req.model, req.horizon, category=req.category, scaler=scaler)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

    return {
        "category": req.category,
        "model": req.model,
        "horizon": req.horizon,
        "forecast": predictions,
    }


@router.post("/compare-models")
def compare_models(req: CompareRequest):
    category_metrics = model_service.metrics.get(req.category)
    if category_metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for category '{req.category}'"
        )

    models_data = {k: v for k, v in category_metrics.items() if k != "best_model"}
    best = category_metrics.get("best_model")

    return {
        "category": req.category,
        "models": models_data,
        "best_model": best,
    }
