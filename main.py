from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers import health, inference
import os

app = FastAPI(title="Disease Detection System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Mount assets directory
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Register Routers
app.include_router(health.router)
app.include_router(inference.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
