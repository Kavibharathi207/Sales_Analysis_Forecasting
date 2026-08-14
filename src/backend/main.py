from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.backend.services.model_service import model_service
from src.backend.api import health, models, metrics, forecast


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_all()
    yield

app = FastAPI(
    title="Pharma Sales Forecasting API",
    description="Model serving API for pharmaceutical sales forecasting",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(metrics.router)
app.include_router(forecast.router)
