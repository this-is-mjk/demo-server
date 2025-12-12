from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers import health, inference
import os

app = FastAPI(
    title="Jaundice Detection API",
    description="AI-powered jaundice detection using EfficientNet models for face and eye analysis",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Mount assets directory
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Register Routers
app.include_router(health.router, tags=["Health"])
app.include_router(inference.router, tags=["Inference"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "inference": "/infer",
            "demo_inference": "/demo/infer"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
