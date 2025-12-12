"""
Model Engine for Jaundice Detection
Handles loading and inference with trained EfficientNet models.
"""

import asyncio
from typing import Dict, Optional
from pathlib import Path

import torch

from config import FACE_MODEL_PATH, EYES_MODEL_PATH, DEVICE
from jaundice_detection.predictor import JaundicePredictor


class ModelEngine:
    """Engine for running jaundice detection models."""
    
    def __init__(self):
        """Initialize and load the jaundice detection models."""
        self.predictor: Optional[JaundicePredictor] = None
        self._load_models()
    
    def _load_models(self):
        """Load face and eyes models."""
        print("Loading jaundice detection models...")
        
        # Check if model files exist
        face_exists = Path(FACE_MODEL_PATH).exists()
        eyes_exists = Path(EYES_MODEL_PATH).exists()
        
        if not face_exists and not eyes_exists:
            print("Warning: No model weights found. Please ensure models are in jaundice_detection/weights/")
            print(f"  Expected face model: {FACE_MODEL_PATH}")
            print(f"  Expected eyes model: {EYES_MODEL_PATH}")
            self.predictor = None
            return
        
        try:
            self.predictor = JaundicePredictor(
                face_model_path=FACE_MODEL_PATH if face_exists else None,
                eyes_model_path=EYES_MODEL_PATH if eyes_exists else None,
                device=DEVICE
            )
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.predictor = None
    
    async def predict_face(self, image_bytes: bytes) -> Dict:
        """
        Run face jaundice detection.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary with prediction results
        """
        if self.predictor is None or not self.predictor.is_loaded('face'):
            return {
                'prediction': 'unknown',
                'is_jaundice': False,
                'confidence': 0.0,
                'probabilities': {'jaundice': 0.0, 'normal': 0.0},
                'model_type': 'face',
                'error': 'Face model not loaded'
            }
        
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.predictor.predict(image_bytes, 'face')
        )
        return result
    
    async def predict_eyes(self, image_bytes: bytes) -> Dict:
        """
        Run eye/sclera jaundice detection.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary with prediction results
        """
        if self.predictor is None or not self.predictor.is_loaded('eyes'):
            return {
                'prediction': 'unknown',
                'is_jaundice': False,
                'confidence': 0.0,
                'probabilities': {'jaundice': 0.0, 'normal': 0.0},
                'model_type': 'eyes',
                'error': 'Eyes model not loaded'
            }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.predictor.predict(image_bytes, 'eyes')
        )
        return result
    
    async def predict_combined(
        self,
        face_image_bytes: Optional[bytes] = None,
        eyes_image_bytes: Optional[bytes] = None
    ) -> Dict:
        """
        Run combined jaundice detection using both models.
        
        Args:
            face_image_bytes: Face image data
            eyes_image_bytes: Eye image data
            
        Returns:
            Dictionary with combined prediction results
        """
        if self.predictor is None:
            return {
                'combined_prediction': 'unknown',
                'is_jaundice': False,
                'combined_confidence': 0.0,
                'combined_probabilities': {'jaundice': 0.0, 'normal': 0.0},
                'individual_results': {},
                'error': 'Models not loaded'
            }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.predictor.predict_combined(face_image_bytes, eyes_image_bytes)
        )
        return result
    
    def get_status(self) -> Dict:
        """Get model loading status."""
        if self.predictor is None:
            return {
                'status': 'not_loaded',
                'loaded_models': [],
                'device': str(DEVICE)
            }
        
        return {
            'status': 'loaded',
            'loaded_models': self.predictor.get_loaded_models(),
            'device': str(self.predictor.device)
        }


# Global model engine instance
model_engine = ModelEngine()
