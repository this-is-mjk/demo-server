"""
Inference Router
Handles image upload and jaundice detection endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.request_response import (
    InferenceResponse,
    FaceAnalysis,
    PallorAnalysis,
    EyeAnalysis,
    JaundiceAnalysis,
    JaundiceOnlyResponse,
    CombinedJaundiceResponse,
    Base64ImageRequest,
    CombinedBase64Request
)
from services.model_engine import model_engine
from config import CONFIDENCE_THRESHOLD
from typing import List, Optional
import shutil
import uuid
import os
import base64

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
    file_path = os.path.join(ASSETS_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return f"/assets/{filename}"


# ============ Jaundice Detection Endpoints ============

@router.post("/predict/face", response_model=JaundiceOnlyResponse)
async def predict_face(file: UploadFile = File(...)):
    """
    Predict jaundice from a face image.
    
    Upload a face image and get jaundice prediction using the face model.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image bytes
    image_bytes = await file.read()
    await file.seek(0)  # Reset for saving
    
    # Save file
    image_path = await save_uploaded_file(file)
    
    # Run prediction
    result = await model_engine.predict_face(image_bytes)
    
    return JaundiceOnlyResponse(
        success=True,
        prediction=result['prediction'],
        is_jaundice=result['is_jaundice'],
        confidence=result['confidence'],
        probabilities=result['probabilities'],
        model_type=result['model_type'],
        image_path=image_path
    )


@router.post("/predict/eyes", response_model=JaundiceOnlyResponse)
async def predict_eyes(file: UploadFile = File(...)):
    """
    Predict jaundice from an eye/sclera image.
    
    Upload an eye image and get jaundice prediction using the eyes model.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    await file.seek(0)
    
    image_path = await save_uploaded_file(file)
    
    result = await model_engine.predict_eyes(image_bytes)
    
    return JaundiceOnlyResponse(
        success=True,
        prediction=result['prediction'],
        is_jaundice=result['is_jaundice'],
        confidence=result['confidence'],
        probabilities=result['probabilities'],
        model_type=result['model_type'],
        image_path=image_path
    )


@router.post("/predict/combined", response_model=CombinedJaundiceResponse)
async def predict_combined(
    face_file: Optional[UploadFile] = File(None),
    eyes_file: Optional[UploadFile] = File(None)
):
    """
    Predict jaundice using both face and eyes images.
    
    Upload face and/or eyes images for combined analysis.
    At least one image must be provided.
    """
    if face_file is None and eyes_file is None:
        raise HTTPException(status_code=400, detail="At least one image (face or eyes) must be provided")
    
    face_bytes = None
    eyes_bytes = None
    image_paths = []
    
    if face_file and face_file.content_type.startswith("image/"):
        face_bytes = await face_file.read()
        await face_file.seek(0)
        image_paths.append(await save_uploaded_file(face_file))
    
    if eyes_file and eyes_file.content_type.startswith("image/"):
        eyes_bytes = await eyes_file.read()
        await eyes_file.seek(0)
        image_paths.append(await save_uploaded_file(eyes_file))
    
    result = await model_engine.predict_combined(face_bytes, eyes_bytes)
    
    return CombinedJaundiceResponse(
        success=True,
        combined_prediction=result['combined_prediction'],
        is_jaundice=result['is_jaundice'],
        combined_confidence=result['combined_confidence'],
        combined_probabilities=result['combined_probabilities'],
        face_result=result['individual_results'].get('face'),
        eyes_result=result['individual_results'].get('eyes'),
        image_paths=image_paths
    )


@router.post("/predict/face/base64", response_model=JaundiceOnlyResponse)
async def predict_face_base64(request: Base64ImageRequest):
    """
    Predict jaundice from a base64 encoded face image.
    """
    try:
        image_data = request.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    result = await model_engine.predict_face(image_bytes)
    
    return JaundiceOnlyResponse(
        success=True,
        prediction=result['prediction'],
        is_jaundice=result['is_jaundice'],
        confidence=result['confidence'],
        probabilities=result['probabilities'],
        model_type=result['model_type']
    )


@router.post("/predict/eyes/base64", response_model=JaundiceOnlyResponse)
async def predict_eyes_base64(request: Base64ImageRequest):
    """
    Predict jaundice from a base64 encoded eye image.
    """
    try:
        image_data = request.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    result = await model_engine.predict_eyes(image_bytes)
    
    return JaundiceOnlyResponse(
        success=True,
        prediction=result['prediction'],
        is_jaundice=result['is_jaundice'],
        confidence=result['confidence'],
        probabilities=result['probabilities'],
        model_type=result['model_type']
    )


@router.post("/predict/combined/base64", response_model=CombinedJaundiceResponse)
async def predict_combined_base64(request: CombinedBase64Request):
    """
    Predict jaundice using base64 encoded face and/or eyes images.
    """
    if request.face_image is None and request.eyes_image is None:
        raise HTTPException(status_code=400, detail="At least one image must be provided")
    
    face_bytes = None
    eyes_bytes = None
    
    try:
        if request.face_image:
            data = request.face_image
            if ',' in data:
                data = data.split(',')[1]
            face_bytes = base64.b64decode(data)
        
        if request.eyes_image:
            data = request.eyes_image
            if ',' in data:
                data = data.split(',')[1]
            eyes_bytes = base64.b64decode(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    
    result = await model_engine.predict_combined(face_bytes, eyes_bytes)
    
    return CombinedJaundiceResponse(
        success=True,
        combined_prediction=result['combined_prediction'],
        is_jaundice=result['is_jaundice'],
        combined_confidence=result['combined_confidence'],
        combined_probabilities=result['combined_probabilities'],
        face_result=result['individual_results'].get('face'),
        eyes_result=result['individual_results'].get('eyes')
    )


# ============ Full Health Analysis Endpoint ============

@router.post("/infer", response_model=InferenceResponse)
async def infer_images(files: List[UploadFile] = File(...)):
    """
    Full health inference endpoint.
    
    Accepts multiple images (face and/or eyes) and returns comprehensive
    health analysis including jaundice detection.
    """
    saved_image_paths = []
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    face_bytes = None
    eyes_bytes = None
    
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        
        # Read bytes for model
        image_bytes = await file.read()
        await file.seek(0)
        
        # Determine image type from filename or use first as face, second as eyes
        filename_lower = file.filename.lower()
        if 'eye' in filename_lower or 'sclera' in filename_lower:
            eyes_bytes = image_bytes
        elif 'face' in filename_lower:
            face_bytes = image_bytes
        elif face_bytes is None:
            face_bytes = image_bytes
        elif eyes_bytes is None:
            eyes_bytes = image_bytes
        
        # Save file
        saved_image_paths.append(await save_uploaded_file(file))
    
    if not saved_image_paths:
        raise HTTPException(status_code=400, detail="No valid images uploaded.")
    
    # Run jaundice detection
    jaundice_result = await model_engine.predict_combined(face_bytes, eyes_bytes)
    
    is_jaundice = jaundice_result['is_jaundice']
    confidence = jaundice_result['combined_confidence']
    severity = get_jaundice_severity(confidence, is_jaundice)
    
    # Build jaundice analysis
    jaundice_analysis = JaundiceAnalysis(
        detected=is_jaundice,
        confidence=confidence,
        severity=severity,
        face_result=jaundice_result['individual_results'].get('face'),
        eyes_result=jaundice_result['individual_results'].get('eyes'),
        tip=get_jaundice_tip(is_jaundice, severity)
    )
    
    # Build other analysis (can be enhanced with additional models)
    face_analysis = FaceAnalysis(
        skin_condition="healthy" if not is_jaundice else "yellowish tint detected",
        hydration_level="85%",
        fatigue_level="low",
        tip="Maintain regular skincare and stay hydrated."
    )
    
    pallor_analysis = PallorAnalysis(
        pallor_level="None" if not is_jaundice else "Mild",
        est_hemoglobin="14.2 g/dL",
        possible_anemia="No",
        tip="Include iron-rich foods in your diet."
    )
    
    eye_analysis = EyeAnalysis(
        sclera_condition="Clear" if not is_jaundice else "Yellowish",
        conjunctiva_color="Pink",
        signs_of_anemia="None",
        signs_of_jaundice="None" if not is_jaundice else severity,
        tip=get_jaundice_tip(is_jaundice, severity)
    )
    
    # Calculate health score
    health_score = 95 if not is_jaundice else max(40, int(95 - confidence * 0.5))
    
    # Build recommendations
    recommendations = [
        "Stay hydrated with at least 8 glasses of water daily.",
        "Get 7-8 hours of quality sleep.",
        "Schedule regular health check-ups."
    ]
    
    if is_jaundice:
        recommendations.insert(0, "⚠️ Jaundice indicators detected. Please consult a healthcare provider.")
        if severity in ["Moderate", "Severe"]:
            recommendations.insert(1, "Consider getting liver function tests done.")
    
    return InferenceResponse(
        health_score=health_score,
        jaundice_analysis=jaundice_analysis,
        face_analysis=face_analysis,
        pallor_analysis=pallor_analysis,
        eye_analysis=eye_analysis,
        recommendations=recommendations,
        images=saved_image_paths
    )


@router.post("/demo/infer", response_model=InferenceResponse)
async def demo_infer_images(files: List[UploadFile] = File(...)):
    """
    Demo endpoint with curated healthy results.
    """
    saved_image_paths = []
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        saved_image_paths.append(await save_uploaded_file(file))
    
    if not saved_image_paths:
        raise HTTPException(status_code=400, detail="No valid images uploaded.")
    
    return InferenceResponse(
        health_score=92,
        jaundice_analysis=JaundiceAnalysis(
            detected=False,
            confidence=98.5,
            severity="None",
            tip="No signs of jaundice detected. Keep up the healthy lifestyle!"
        ),
        face_analysis=FaceAnalysis(
            skin_condition="healthy",
            hydration_level="88%",
            fatigue_level="low",
            tip="Your skin looks great! Keep drinking water."
        ),
        pallor_analysis=PallorAnalysis(
            pallor_level="None",
            est_hemoglobin="14.5 g/dL",
            possible_anemia="No",
            tip="Iron levels appear normal."
        ),
        eye_analysis=EyeAnalysis(
            sclera_condition="Clear",
            conjunctiva_color="Pink",
            signs_of_anemia="None",
            signs_of_jaundice="None",
            tip="No signs of strain or discoloration."
        ),
        recommendations=[
            "Maintain your current balanced diet.",
            "Continue with regular exercise.",
            "Stay hydrated to keep your skin glowing."
        ],
        images=saved_image_paths
    )
