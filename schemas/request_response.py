from pydantic import BaseModel
from typing import Dict, List, Optional

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

class AudioAnalysis(BaseModel):
    pitch_hz: float
    jitter_percent: float
    shimmer_percent: float
    parkinsons_detected: bool
    confidence: float
    warning: Optional[str] = None
    error: Optional[str] = None

class InferenceResponse(BaseModel):
    health_score: int
    face_analysis: Optional[FaceAnalysis] = None
    pallor_analysis: Optional[PallorAnalysis] = None
    eye_analysis: Optional[EyeAnalysis] = None
    audio_analysis: Optional[AudioAnalysis] = None
    recommendations: List[str]
    images: List[str]

class HealthStatus(BaseModel):
    status: str
    version: str
