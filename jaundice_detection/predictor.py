"""
Jaundice Predictor
Handles model loading and inference for face and eye images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
import base64
from io import BytesIO

from .model import JaundiceClassifier


class JaundicePredictor:
    """Predictor for jaundice detection using trained EfficientNet models."""
    
    def __init__(
        self,
        face_model_path: Optional[str] = None,
        eyes_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the predictor with model paths.
        
        Args:
            face_model_path: Path to the trained face model checkpoint
            eyes_model_path: Path to the trained eyes model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.models: Dict[str, nn.Module] = {}
        self.class_names = ['jaundice', 'normal']
        
        # Load models if paths provided
        if face_model_path and Path(face_model_path).exists():
            self.models['face'] = self._load_model(face_model_path)
            print(f"✓ Face model loaded from: {face_model_path}")
        
        if eyes_model_path and Path(eyes_model_path).exists():
            self.models['eyes'] = self._load_model(eyes_model_path)
            print(f"✓ Eyes model loaded from: {eyes_model_path}")
        
        if not self.models:
            print("⚠ Warning: No models loaded. Predictor will return mock results.")
        
        print(f"✓ Using device: {self.device}")
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load a model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model = JaundiceClassifier(
            model_name=checkpoint.get('model_name', 'efficientnet_b0'),
            num_classes=2,
            pretrained=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image: Union[Image.Image, bytes, str]) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: PIL Image, bytes, or base64 string
            
        Returns:
            Preprocessed tensor ready for inference
        """
        if isinstance(image, str):
            # Assume base64 encoded
            if ',' in image:
                image = image.split(',')[1]
            image_data = base64.b64decode(image)
            image = Image.open(BytesIO(image_data))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(
        self,
        image: Union[Image.Image, bytes, str],
        model_type: str = 'face'
    ) -> Dict:
        """
        Make a prediction on a single image.
        
        Args:
            image: Input image (PIL, bytes, or base64)
            model_type: 'face' or 'eyes'
            
        Returns:
            Dictionary with prediction results
        """
        if model_type not in self.models:
            # Return mock result if model not loaded
            return {
                'prediction': 'unknown',
                'is_jaundice': False,
                'confidence': 0.0,
                'probabilities': {'jaundice': 0.0, 'normal': 0.0},
                'model_type': model_type,
                'error': f'Model {model_type} not loaded'
            }
        
        model = self.models[model_type]
        
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            prob_jaundice = probabilities[0, 0].item()
            prob_normal = probabilities[0, 1].item()
            
            is_jaundice = prob_jaundice > prob_normal
            confidence = prob_jaundice if is_jaundice else prob_normal
            prediction = 'jaundice' if is_jaundice else 'normal'
        
        return {
            'prediction': prediction,
            'is_jaundice': is_jaundice,
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'jaundice': round(prob_jaundice * 100, 2),
                'normal': round(prob_normal * 100, 2)
            },
            'model_type': model_type
        }
    
    def predict_combined(
        self,
        face_image: Optional[Union[Image.Image, bytes, str]] = None,
        eyes_image: Optional[Union[Image.Image, bytes, str]] = None,
        face_weight: float = 0.6,
        eyes_weight: float = 0.4
    ) -> Dict:
        """
        Make a combined prediction using both face and eyes models.
        
        Args:
            face_image: Face image input
            eyes_image: Eyes image input
            face_weight: Weight for face model prediction
            eyes_weight: Weight for eyes model prediction
            
        Returns:
            Dictionary with combined prediction results
        """
        results = {}
        
        # Get individual predictions
        if face_image is not None and 'face' in self.models:
            results['face'] = self.predict(face_image, 'face')
        
        if eyes_image is not None and 'eyes' in self.models:
            results['eyes'] = self.predict(eyes_image, 'eyes')
        
        if not results:
            return {
                'combined_prediction': 'unknown',
                'is_jaundice': False,
                'combined_confidence': 0.0,
                'combined_probabilities': {'jaundice': 0.0, 'normal': 0.0},
                'individual_results': {},
                'error': 'No valid predictions available'
            }
        
        # Calculate weighted combination
        total_weight = 0
        weighted_jaundice = 0
        weighted_normal = 0
        
        if 'face' in results:
            weighted_jaundice += results['face']['probabilities']['jaundice'] * face_weight
            weighted_normal += results['face']['probabilities']['normal'] * face_weight
            total_weight += face_weight
        
        if 'eyes' in results:
            weighted_eyes = eyes_weight if 'face' in results else 1.0
            weighted_jaundice += results['eyes']['probabilities']['jaundice'] * weighted_eyes
            weighted_normal += results['eyes']['probabilities']['normal'] * weighted_eyes
            total_weight += weighted_eyes
        
        # Normalize
        if total_weight > 0:
            weighted_jaundice /= total_weight
            weighted_normal /= total_weight
        
        is_jaundice = weighted_jaundice > weighted_normal
        confidence = weighted_jaundice if is_jaundice else weighted_normal
        prediction = 'jaundice' if is_jaundice else 'normal'
        
        return {
            'combined_prediction': prediction,
            'is_jaundice': is_jaundice,
            'combined_confidence': round(confidence, 2),
            'combined_probabilities': {
                'jaundice': round(weighted_jaundice, 2),
                'normal': round(weighted_normal, 2)
            },
            'individual_results': results
        }
    
    def is_loaded(self, model_type: str) -> bool:
        """Check if a specific model is loaded."""
        return model_type in self.models
    
    def get_loaded_models(self) -> list:
        """Get list of loaded model types."""
        return list(self.models.keys())
