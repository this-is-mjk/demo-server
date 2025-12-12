# Jaundice Detection API

## Overview
The Jaundice Detection API is a FastAPI-based backend service powered by EfficientNet deep learning models. It analyzes face and eye images to detect signs of jaundice with high accuracy.

## Features
- **AI-Powered Detection**: Uses trained EfficientNet-B0 models for jaundice classification
- **Dual Model System**: Separate models for face and eye analysis
- **Combined Predictions**: Weighted ensemble of both models for improved accuracy
- **Multiple Input Formats**: Supports file upload and base64 encoded images
- **Comprehensive Health Analysis**: Full health report with jaundice detection integrated
- **CORS Enabled**: Ready for frontend integration

## Model Performance
| Model | Test Accuracy | ROC AUC |
|-------|--------------|---------|
| Face | 92% | - |
| Eyes | 86% | 96.43% |

## Tech Stack
- **Framework**: FastAPI (Python)
- **Deep Learning**: PyTorch, EfficientNet (timm)
- **Image Processing**: Pillow, OpenCV
- **Server**: Uvicorn

## Project Structure
```
demo-server/
├── main.py                    # FastAPI application entry
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── routers/
│   ├── health.py             # Health check endpoint
│   └── inference.py          # Prediction endpoints
├── schemas/
│   └── request_response.py   # Pydantic models
├── services/
│   ├── model_engine.py       # Model loading & inference
│   └── decision_maker.py     # Prediction aggregation
├── jaundice_detection/       # ML module
│   ├── model.py              # EfficientNet architecture
│   ├── predictor.py          # Inference wrapper
│   ├── train.py              # Training script
│   ├── prepare_dataset.py    # Dataset utilities
│   ├── extract_eyes.py       # Eye extraction
│   └── weights/              # Trained model weights
│       ├── face/best_model.pth
│       └── eyes/best_model.pth
└── assets/                   # Uploaded images
```

## Installation & Setup

### Prerequisites
- Python 3.9+
- CUDA (optional, for GPU inference)

### Local Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd demo-server
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure model weights are in place:
   ```
   jaundice_detection/weights/face/best_model.pth
   jaundice_detection/weights/eyes/best_model.pth
   ```

4. Run the server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

   The server will start at `http://localhost:8000`

### Docker Setup
```bash
docker build -t jaundice-api .
docker run -p 8000:8000 jaundice-api
```

## API Endpoints

### Health Check
- **URL**: `GET /health`
- **Response**:
  ```json
  {
    "status": "ok",
    "version": "1.0.0",
    "models": {
      "status": "loaded",
      "loaded_models": ["face", "eyes"],
      "device": "cuda"
    }
  }
  ```

### Face Jaundice Detection
- **URL**: `POST /predict/face`
- **Body**: `multipart/form-data` with `file` (image)
- **Response**:
  ```json
  {
    "success": true,
    "prediction": "jaundice",
    "is_jaundice": true,
    "confidence": 95.5,
    "probabilities": {"jaundice": 95.5, "normal": 4.5},
    "model_type": "face",
    "image_path": "/assets/uuid.jpg"
  }
  ```

### Eye Jaundice Detection
- **URL**: `POST /predict/eyes`
- **Body**: `multipart/form-data` with `file` (eye image)

### Combined Detection
- **URL**: `POST /predict/combined`
- **Body**: `multipart/form-data` with `face_file` and/or `eyes_file`
- **Response**:
  ```json
  {
    "success": true,
    "combined_prediction": "jaundice",
    "is_jaundice": true,
    "combined_confidence": 92.3,
    "combined_probabilities": {"jaundice": 92.3, "normal": 7.7},
    "face_result": {...},
    "eyes_result": {...}
  }
  ```

### Base64 Endpoints
- `POST /predict/face/base64` - Face prediction with base64 image
- `POST /predict/eyes/base64` - Eyes prediction with base64 image
- `POST /predict/combined/base64` - Combined prediction with base64 images

### Full Health Analysis
- **URL**: `POST /infer`
- **Body**: `multipart/form-data` with `files` (multiple images)
- **Response**: Complete health analysis including jaundice detection, face analysis, pallor analysis, and eye analysis

## Retraining Models

To retrain the jaundice detection models:

1. **Prepare your dataset**:
   ```bash
   cd jaundice_detection
   python prepare_dataset.py --raw-dir /path/to/raw/images --output-dir dataset
   ```

2. **Train the model**:
   ```bash
   python train.py --model-type face --data-dir dataset --epochs 30
   python train.py --model-type eyes --data-dir dataset_eyes --epochs 30
   ```

3. Move the new weights to `weights/face/` and `weights/eyes/`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Device for inference (`cuda` or `cpu`) |
| `ENV` | `development` | Environment mode |

## License
MIT
