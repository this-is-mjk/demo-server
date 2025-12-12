# Configuration & Environment variables
import os
from pathlib import Path

# Environment
ENV = os.getenv("ENV", "development")

# Base paths
BASE_DIR = Path(__file__).parent
JAUNDICE_DETECTION_DIR = BASE_DIR / "jaundice_detection"
WEIGHTS_DIR = JAUNDICE_DETECTION_DIR / "weights"

# Model paths
FACE_MODEL_PATH = str(WEIGHTS_DIR / "face" / "best_model.pth")
EYES_MODEL_PATH = str(WEIGHTS_DIR / "eyes" / "best_model.pth")

# Model configuration
MODEL_CONFIG = {
    "face": {
        "path": FACE_MODEL_PATH,
        "weight": 0.6,  # Weight for combined prediction
        "description": "EfficientNet-B0 for face jaundice detection"
    },
    "eyes": {
        "path": EYES_MODEL_PATH,
        "weight": 0.4,  # Weight for combined prediction
        "description": "EfficientNet-B0 for eye sclera jaundice detection"
    }
}

# Inference settings
DEVICE = os.getenv("DEVICE", "cuda")  # 'cuda' or 'cpu'
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for positive jaundice detection

# Assets directory for uploaded images
ASSETS_DIR = BASE_DIR / "assets"
