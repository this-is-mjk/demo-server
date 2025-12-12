from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.request_response import (
    InferenceResponse,
    FaceAnalysis,
    PallorAnalysis,
    EyeAnalysis,
    AudioAnalysis
)
from services.audio_engine import audio_engine
from typing import List, Optional
import shutil
import uuid
import os
import random

router = APIRouter()

ASSETS_DIR = "assets"


@router.post("/infer", response_model=InferenceResponse)
async def infer_images(
    files: List[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    saved_image_paths = []
    
    # Ensure assets directory exists (redundant if main.py does it, but safe)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    if files:
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

    # Mock Analysis Data (conditionally)
    face_analysis = None
    pallor_analysis = None
    eye_analysis = None

    if saved_image_paths:
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
    
    audio_analysis = None
    if audio:
        # Save audio file
        audio_ext = audio.filename.split(".")[-1] if "." in audio.filename else "wav"
        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.{audio_ext}"
        audio_path = os.path.join(ASSETS_DIR, audio_filename)
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        saved_image_paths.append(f"/assets/{audio_filename}")
        
        # Analyze audio
        audio_analysis_data = audio_engine.analyze_audio(audio_path)
        if audio_analysis_data and "error" not in audio_analysis_data and "warning" not in audio_analysis_data:
             audio_analysis = AudioAnalysis(**audio_analysis_data)
        elif audio_analysis_data:
             # handle error/warning cases gracefully or just return partial info
             audio_analysis = AudioAnalysis(
                 pitch_hz=0, jitter_percent=0, shimmer_percent=0, 
                 parkinsons_detected=False, confidence=0, 
                 **audio_analysis_data
             )

    if not saved_image_paths and not audio_analysis:
        raise HTTPException(status_code=400, detail="No valid input provided (images or audio).")
    
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
        audio_analysis=audio_analysis,
        images=saved_image_paths
    )

@router.post("/demo/infer", response_model=InferenceResponse)
async def demo_infer_images(
    files: List[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Demo endpoint returning curated realistic dummy data.
    """
    saved_image_paths = []
    
    # Ensure assets directory exists
    os.makedirs(ASSETS_DIR, exist_ok=True)

    if files:
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
        
    if audio:
        audio_ext = audio.filename.split(".")[-1] if "." in audio.filename else "wav"
        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.{audio_ext}"
        audio_path = os.path.join(ASSETS_DIR, audio_filename)
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        saved_image_paths.append(f"/assets/{audio_filename}")


    if not saved_image_paths and not audio:
         raise HTTPException(status_code=400, detail="No valid input provided.")

    # Curated "Real-looking" Dummy Data for Demo (Conditionally)
    face_analysis = None
    pallor_analysis = None
    eye_analysis = None

    # Check specifically if images were saved (saved_image_paths contains images or audio, 
    # but we only want image analysis if images were uploaded. 
    # Distinguishing logic: check if any image-like path is in saved_image_paths OR simply check 'files' input.
    # 'files' input is cleaner.
    if files:
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
    
    audio_analysis = None
    if audio:
        audio_analysis = AudioAnalysis(
            pitch_hz=120.5,
            jitter_percent=0.45,
            shimmer_percent=1.2,
            parkinsons_detected=False,
            confidence=98.5
        )
    
    return InferenceResponse(
        health_score=92,
        face_analysis=face_analysis,
        pallor_analysis=pallor_analysis,
        eye_analysis=eye_analysis,
        audio_analysis=audio_analysis,
        recommendations=[
            "Maintain your current balanced diet.",
            "Continue with regular exercise.",
            "Stay hydrated to keep your skin glowing."
        ],
        images=saved_image_paths
    )
