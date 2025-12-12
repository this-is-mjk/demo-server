from pydantic import BaseModel
from typing import Dict, List, Optional


# ============ Jaundice Detection Schemas ============

class JaundicePrediction(BaseModel):
    """Single model prediction result."""
    prediction: str  # 'jaundice' or 'normal'
    is_jaundice: bool
    confidence: float  # Confidence percentage (0-100)
    probabilities: Dict[str, float]  # {'jaundice': x, 'normal': y}
    model_type: str  # 'face' or 'eyes'


class CombinedJaundicePrediction(BaseModel):
    """Combined prediction from multiple models."""
    combined_prediction: str
    is_jaundice: bool
    combined_confidence: float
    combined_probabilities: Dict[str, float]
    individual_results: Dict[str, JaundicePrediction]


class JaundiceAnalysis(BaseModel):
    """Detailed jaundice analysis for health report."""
    detected: bool
    confidence: float
    severity: str  # 'None', 'Mild', 'Moderate', 'Severe'
    face_result: Optional[JaundicePrediction] = None
    eyes_result: Optional[JaundicePrediction] = None
    tip: str


# ============ Original Health Analysis Schemas ============

class FaceAnalysis(BaseModel):
    skin_condition: str
    hydration_level: str
    fatigue_level: str
    tip: str


class PallorAnalysis(BaseModel):
    pallor_level: str
    est_hemoglobin: str
    possible_anemia: str
    tip: str


class EyeAnalysis(BaseModel):
    sclera_condition: str
    conjunctiva_color: str
    signs_of_anemia: str
    signs_of_jaundice: str
    tip: str


# ============ Response Schemas ============

class InferenceResponse(BaseModel):
    """Full health inference response."""
    health_score: int
    jaundice_analysis: Optional[JaundiceAnalysis] = None
    face_analysis: FaceAnalysis
    pallor_analysis: PallorAnalysis
    eye_analysis: EyeAnalysis
    recommendations: List[str]
    images: List[str]


class JaundiceOnlyResponse(BaseModel):
    """Response for jaundice-only detection endpoints."""
    success: bool
    prediction: str
    is_jaundice: bool
    confidence: float
    probabilities: Dict[str, float]
    model_type: str
    image_path: Optional[str] = None


class CombinedJaundiceResponse(BaseModel):
    """Response for combined jaundice detection."""
    success: bool
    combined_prediction: str
    is_jaundice: bool
    combined_confidence: float
    combined_probabilities: Dict[str, float]
    face_result: Optional[Dict] = None
    eyes_result: Optional[Dict] = None
    image_paths: List[str] = []


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    version: str
    models: Optional[Dict] = None


# ============ Request Schemas ============

class Base64ImageRequest(BaseModel):
    """Request with base64 encoded image."""
    image: str  # Base64 encoded image


class CombinedBase64Request(BaseModel):
    """Request with both face and eyes images as base64."""
    face_image: Optional[str] = None
    eyes_image: Optional[str] = None
