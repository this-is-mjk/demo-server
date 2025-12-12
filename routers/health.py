from fastapi import APIRouter
from schemas.request_response import HealthStatus

router = APIRouter()

@router.get("/health", response_model=HealthStatus)
async def health_check():
    return HealthStatus(status="ok", version="1.0.0")
