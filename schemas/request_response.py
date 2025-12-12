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

class InferenceResponse(BaseModel):
    health_score: int
    face_analysis: FaceAnalysis
    pallor_analysis: PallorAnalysis
    eye_analysis: EyeAnalysis
    recommendations: List[str]
    images: List[str]

class HealthStatus(BaseModel):
    status: str
    version: str
