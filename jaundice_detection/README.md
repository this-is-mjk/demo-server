# Jaundice Detection Module

This module contains the EfficientNet-based models for detecting jaundice from face and eye images.

## Structure

```
jaundice_detection/
├── __init__.py           # Module initialization
├── model.py              # EfficientNet classifier architecture
├── predictor.py          # Inference wrapper class
├── train.py              # Training script
├── prepare_dataset.py    # Dataset preparation utility
├── extract_eyes.py       # Eye region extraction from faces
└── weights/              # Trained model weights
    ├── face/
    │   ├── best_model.pth
    │   └── config.json
    └── eyes/
        ├── best_model.pth
        └── config.json
```

## Model Details

### Face Model
- **Architecture**: EfficientNet-B0 with custom classifier head
- **Input**: 224x224 RGB face images
- **Output**: Binary classification (jaundice/normal)
- **Test Accuracy**: 92%

### Eyes Model
- **Architecture**: EfficientNet-B0 with custom classifier head
- **Input**: 224x224 RGB eye/sclera images
- **Output**: Binary classification (jaundice/normal)
- **Test Accuracy**: 86%
- **ROC AUC**: 96.43%

## Usage

### Inference

```python
from jaundice_detection import JaundicePredictor

# Initialize predictor
predictor = JaundicePredictor(
    face_model_path='jaundice_detection/weights/face/best_model.pth',
    eyes_model_path='jaundice_detection/weights/eyes/best_model.pth'
)

# Predict from image bytes
result = predictor.predict(image_bytes, model_type='face')
print(result)
# {'prediction': 'jaundice', 'is_jaundice': True, 'confidence': 95.5, ...}

# Combined prediction
result = predictor.predict_combined(face_image, eyes_image)
```

### Retraining

1. **Prepare dataset**:
```bash
python -m jaundice_detection.prepare_dataset \
    --raw-dir raw_faces \
    --output-dir dataset_faces
```

2. **Train model**:
```bash
python -m jaundice_detection.train \
    --model-type face \
    --data-dir dataset_faces \
    --output-dir weights \
    --epochs 30
```

3. **Extract eyes** (optional):
```bash
python -m jaundice_detection.extract_eyes \
    --input-dir raw_faces \
    --output-dir raw_eyes
```

## Dataset Structure

Expected raw data structure:
```
raw/
├── jaundice/
│   ├── image1.jpg
│   └── ...
└── normal/
    ├── image1.jpg
    └── ...
```

After preparation:
```
dataset/
├── train/
│   ├── jaundice/
│   └── normal/
├── val/
│   ├── jaundice/
│   └── normal/
└── test/
    ├── jaundice/
    └── normal/
```

## API Endpoints

When integrated with the FastAPI server:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/face` | POST | Jaundice detection from face image |
| `/predict/eyes` | POST | Jaundice detection from eye image |
| `/predict/combined` | POST | Combined analysis using both images |
| `/predict/face/base64` | POST | Face prediction with base64 image |
| `/predict/eyes/base64` | POST | Eyes prediction with base64 image |
| `/predict/combined/base64` | POST | Combined prediction with base64 images |
| `/infer` | POST | Full health analysis with jaundice detection |

## Requirements

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- Pillow >= 9.0.0
- OpenCV (for eye extraction)
