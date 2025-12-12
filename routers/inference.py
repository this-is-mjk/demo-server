from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.request_response import (
    InferenceResponse,
    FaceAnalysis,
    PallorAnalysis,
    EyeAnalysis
)
from typing import List
import shutil
import uuid
import os
import random

router = APIRouter()

ASSETS_DIR = "assets"

@router.post("/infer", response_model=InferenceResponse)
async def infer_images(files: List[UploadFile] = File(...)):
    saved_image_paths = []
    
    # Ensure assets directory exists (redundant if main.py does it, but safe)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    for file in files:
        if not file.content_type.startswith("image/"):
            continue # Skip non-image files or raise error
            
        # Generate unique filename
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.{file_extension}"
        file_path = os.path.join(ASSETS_DIR, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Add to list of accessible paths (assuming served at /assets/)
        saved_image_paths.append(f"/assets/{filename}")

    if not saved_image_paths:
         raise HTTPException(status_code=400, detail="No valid images uploaded.")

    # Mock Analysis Data (since actual models for these features aren't integrated yet)
    # in a real scenario, we would process the images here.
    
    face_analysis = FaceAnalysis(
        skin_condition=random.choice(["healthy", "moderate", "bad"]),
        hydration_level=f"{random.randint(60, 98)}%",
        fatigue_level=random.choice(["mild", "high", "low"]),
        tip="Maintain a consistent sleep schedule and stay hydrated."
    )
    
    pallor_analysis = PallorAnalysis(
        pallor_level=random.choice(["None", "Mild", "Severe"]),
        est_hemoglobin=f"{random.randint(11, 16)} g/dL",
        possible_anemia="No",
        tip="Consume iron-rich foods like spinach and red meat."
    )
    
    eye_analysis = EyeAnalysis(
        sclera_condition="Clear",
        conjunctiva_color="Pink",
        signs_of_anemia="None",
        signs_of_jaundice="None",
        tip="Rest your eyes every 20 minutes to prevent strain."
    )
    
    return InferenceResponse(
        health_score=random.randint(70, 100),
        face_analysis=face_analysis,
        pallor_analysis=pallor_analysis,
        eye_analysis=eye_analysis,
        recommendations=[
            "Drink at least 8 glasses of water a day.",
            "Get 7-8 hours of sleep.",
            "Schedule a regular check-up."
        ],
        images=saved_image_paths
    )

@router.post("/demo/infer", response_model=InferenceResponse)
async def demo_infer_images(files: List[UploadFile] = File(...)):
    """
    Demo endpoint returning curated realistic dummy data.
    """
    saved_image_paths = []
    
    # Ensure assets directory exists
    os.makedirs(ASSETS_DIR, exist_ok=True)

    for file in files:
        if not file.content_type.startswith("image/"):
            continue 
            
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.{file_extension}"
        file_path = os.path.join(ASSETS_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        saved_image_paths.append(f"/assets/{filename}")

    if not saved_image_paths:
         raise HTTPException(status_code=400, detail="No valid images uploaded.")

    # Curated "Real-looking" Dummy Data for Demo
    face_analysis = FaceAnalysis(
        skin_condition="healthy",
        hydration_level="88%",
        fatigue_level="low",
        tip="Your skin looks great! Keep drinking water."
    )
    
    pallor_analysis = PallorAnalysis(
        pallor_level="None",
        est_hemoglobin="14.5 g/dL",
        possible_anemia="No",
        tip="Iron levels appear normal."
    )
    
    eye_analysis = EyeAnalysis(
        sclera_condition="Clear",
        conjunctiva_color="Pink",
        signs_of_anemia="None",
        signs_of_jaundice="None",
        tip="No signs of strain or discoloration."
    )
    
    return InferenceResponse(
        health_score=92,
        face_analysis=face_analysis,
        pallor_analysis=pallor_analysis,
        eye_analysis=eye_analysis,
        recommendations=[
            "Maintain your current balanced diet.",
            "Continue with regular exercise.",
            "Stay hydrated to keep your skin glowing."
        ],
        images=saved_image_paths
    )
