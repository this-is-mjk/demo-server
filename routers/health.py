from fastapi import APIRouter
from schemas.request_response import HealthStatus
from services.model_engine import model_engine

router = APIRouter()

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint with model status."""
    status = model_engine.get_status()
    return HealthStatus(
        status="ok" if status['status'] == 'loaded' else "degraded",
        version="1.0.0",
        models=status
    )
