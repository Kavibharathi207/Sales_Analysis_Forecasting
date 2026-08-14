"""
main.py — FastAPI entry point for the Sales Analysis backend.

Run:
    uvicorn backend.main:app --reload

Docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from backend.routes import anomaly, whatif, recommendation

app = FastAPI(
    title="Pharma Sales Analysis API",
    description="Anomaly Detection · What-If Analysis · Recommendations",
    version="1.0.0",
)

app.include_router(anomaly.router)
app.include_router(whatif.router)
app.include_router(recommendation.router)


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "message": "Pharma Sales Analysis API is running."}
