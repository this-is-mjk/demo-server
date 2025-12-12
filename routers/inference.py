"""
Inference Router
Handles image upload, jaundice detection, and audio analysis endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.request_response import (
    InferenceResponse,
    JaundiceAnalysis,
    AudioAnalysis
)
from services.model_engine import model_engine
from services.audio_engine import audio_engine
from config import CONFIDENCE_THRESHOLD
from typing import List, Optional
import shutil
import uuid
import os

router = APIRouter()

ASSETS_DIR = "assets"


def get_jaundice_severity(confidence: float, is_jaundice: bool) -> str:
    """Determine severity level based on confidence."""
    if not is_jaundice:
        return "None"
    if confidence >= 95:
        return "Severe"
    elif confidence >= 80:
        return "Moderate"
    elif confidence >= CONFIDENCE_THRESHOLD * 100:
        return "Mild"
    return "None"


def get_jaundice_tip(is_jaundice: bool, severity: str) -> str:
    """Get health tip based on jaundice detection."""
    if not is_jaundice:
        return "No signs of jaundice detected. Maintain a healthy diet rich in fruits and vegetables."
    
    tips = {
        "Mild": "Mild jaundice signs detected. Consider consulting a healthcare provider and stay hydrated.",
        "Moderate": "Moderate jaundice indicators found. Please schedule an appointment with your doctor soon.",
        "Severe": "Significant jaundice signs detected. Seek medical attention promptly for proper evaluation."
    }
    return tips.get(severity, "Please consult a healthcare professional for accurate diagnosis.")


async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return the path."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    file_id = str(uuid.uuid4())
    filename = f"{file_id}.{file_extension}"
    # Ensure unique filename collision avoidance
    file_path = os.path.join(ASSETS_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return f"/assets/{filename}"


# ============ Full Health Analysis Endpoint ============

@router.post("/infer", response_model=InferenceResponse)
async def infer_images(
    files: List[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Full health inference endpoint.
    Accepts multiple images (face and/or eyes) and/or audio.
    Returns comprehensive health analysis including jaundice detection and audio analysis.
    """
    saved_image_paths = []
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # --- 1. Audio Processing ---
    audio_analysis = None
    if audio:
        audio_ext = audio.filename.split(".")[-1] if "." in audio.filename else "wav"
        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.{audio_ext}"
        audio_path = os.path.join(ASSETS_DIR, audio_filename)
        
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        saved_image_paths.append(f"/assets/{audio_filename}")
        
        # Analyze audio
        audio_results = audio_engine.analyze_audio(audio_path)
        if audio_results and "error" not in audio_results and "warning" not in audio_results:
             audio_analysis = AudioAnalysis(**audio_results)
        elif audio_results:
             # handle error/warning cases gracefully
             audio_analysis = AudioAnalysis(
                 pitch_hz=0, jitter_percent=0, shimmer_percent=0, 
                 parkinsons_detected=False, confidence=0, 
                 **audio_results
             )

    # --- 2. Image Processing ---
    face_bytes = None
    eyes_bytes = None
    # --- 2. Image Processing & Jaundice Detection ---
    jaundice_analysis = None
    
    # Aggregation variables
    is_jaundice_detected_any = False
    max_confidence_jaundice = 0.0
    max_confidence_normal = 0.0
    
    # Keep track of individual results to find the "worst" case
    worst_case_result = None
    
    if files:
        for file in files:
            if not file.content_type.startswith("image/"):
                continue
            
            # Read bytes for model
            image_bytes = await file.read()
            await file.seek(0)
            
            # Save file
            saved_image_paths.append(await save_uploaded_file(file))

            # Determine model type based on filename
            filename_lower = file.filename.lower()
            model_type = 'eyes' if ('eye' in filename_lower or 'sclera' in filename_lower) else 'face'
            
            # Run prediction
            if model_type == 'eyes':
                 result = await model_engine.predict_eyes(image_bytes)
            else:
                 result = await model_engine.predict_face(image_bytes)
            
            # Aggregate logic: "If you find issue with some file go ahead and report jaundice"
            if result['is_jaundice']:
                is_jaundice_detected_any = True
                if result['confidence'] > max_confidence_jaundice:
                    max_confidence_jaundice = result['confidence']
                    worst_case_result = result
            else:
                # Track highest confidence normal if no jaundice found yet
                if result['confidence'] > max_confidence_normal:
                    max_confidence_normal = result['confidence']
                    if not is_jaundice_detected_any and worst_case_result is None:
                         worst_case_result = result

    # Final decision logic
    is_jaundice = is_jaundice_detected_any
    confidence = max_confidence_jaundice if is_jaundice else max_confidence_normal
    
    # Fallback if no valid images processed
    if worst_case_result is None and saved_image_paths:
         # Should effectively not happen if images were processed, but safety check
         confidence = 0
         if not is_jaundice: confidence = 0

    severity = get_jaundice_severity(confidence, is_jaundice)

    if worst_case_result:
        jaundice_analysis = JaundiceAnalysis(
            detected=is_jaundice,
            confidence=confidence,
            severity=severity,
            # We map specific result to the schema. 
            # Note: The schema expects 'face_result' and 'eyes_result'. 
            # We populate the relevant one based on what triggered the decision.
            face_result=worst_case_result if worst_case_result['model_type'] == 'face' else None,
            eyes_result=worst_case_result if worst_case_result['model_type'] == 'eyes' else None,
            tip=get_jaundice_tip(is_jaundice, severity)
        )
        

    # Check validity
    if not saved_image_paths and not audio_analysis:
        raise HTTPException(status_code=400, detail="No valid input provided (images or audio).")
    
    # Calculate health score
    # logic: Start at 100.
    # If Jaundice detected: deduct up to 60 points based on confidence (40-100 range).
    # If Parkinson's detected: deduct up to 50 points based on confidence.
    # Minimum score cap at 20.
    current_score = 100

    if is_jaundice:
        # Confidence is 0-100. If 100% confident, deduct 60 points -> score 40.
        deduction = (confidence / 100.0) * 60
        current_score -= deduction

    if audio_analysis and audio_analysis.parkinsons_detected:
        # If Parkinson's detected with 100% confidence, deduct 50 points.
        deduction = (audio_analysis.confidence / 100.0) * 50
        current_score -= deduction

    # Ensure score stays within reasonable bounds [10, 100]
    health_score = int(max(10, current_score))

    # Recommendations
    recommendations = [
        "Stay hydrated with at least 8 glasses of water daily.",
        "Get 7-8 hours of quality sleep.",
        "Schedule regular health check-ups."
    ]
    
    if is_jaundice:
        recommendations.insert(0, "⚠️ Jaundice indicators detected. Please consult a healthcare provider.")
        if severity in ["Moderate", "Severe"]:
            recommendations.insert(1, "Consider getting liver function tests done.")
            
    if audio_analysis and audio_analysis.parkinsons_detected:
        recommendations.insert(0, "⚠️ Audio analysis suggests potential Parkinson's indicators. Please consult a specialist.")

    return InferenceResponse(
        health_score=health_score,
        jaundice_analysis=jaundice_analysis,
        audio_analysis=audio_analysis,
        recommendations=recommendations,
        images=saved_image_paths
    )


@router.post("/demo/infer", response_model=InferenceResponse)
async def demo_infer_images(
    files: List[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Demo endpoint with curated results.
    """
    saved_image_paths = []
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # Demo Audio
    audio_analysis = None
    if audio:
        saved_image_paths.append(await save_uploaded_file(audio))
        audio_analysis = AudioAnalysis(
            pitch_hz=120.5,
            jitter_percent=0.45,
            shimmer_percent=1.2,
            parkinsons_detected=False,
            confidence=98.5
        )

    # Demo Images
    has_images = False
    if files:
        for file in files:
            if not file.content_type.startswith("image/"):
                continue
            saved_image_paths.append(await save_uploaded_file(file))
            has_images = True
            
    if not saved_image_paths and not audio_analysis:
        raise HTTPException(status_code=400, detail="No valid inputs.")

    return InferenceResponse(
        health_score=85, # Demo score: 100 - (15 deduction for partial confidence)
        jaundice_analysis=JaundiceAnalysis(
            detected=False,
            confidence=98.5,
            severity="None",
            tip="No signs of jaundice detected. Keep up the healthy lifestyle!"
        ) if has_images else None,
        audio_analysis=audio_analysis,
        recommendations=[
            "Maintain your current balanced diet.",
            "Continue with regular exercise.",
            "Stay hydrated to keep your skin glowing."
        ],
        images=saved_image_paths
    )
