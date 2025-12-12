# Disease Detection API

## Overview
The **Disease Detection API** is a comprehensive diagnostic system that leverages multi-modal machine learning to detect early signs of disease. It combines computer vision for **Jaundice Detection** and acoustic analysis for **Parkinson's Disease Detection** into a single, high-performance REST API.

## Features
- **Multi-Modal Inference**: Process images (Face/Eyes) and audio (Voice) in a single request.
- **Jaundice Detection**:
  - Uses **EfficientNet-B0** Deep Learning models.
  - Used both detection through skin and eye color, to give complete diagnosis.
  - specialized models for **Face** and **Eyes**.
  - **High Accuracy (92% Face, 86% Eyes)**.
- **Parkinson's Detection**:
  - Uses **Random Forest Classifier** on acoustic biomarkers.
  - Analyzes **16 distinct vocal features** (Jitter, Shimmer, HNR, etc.).
  - **97.6% Accuracy** on test data.
- **Real-Time Analysis**: Fast inference on CPU or GPU/MPS.
- **Health Score Engine**: integrated scoring system that quantifies risk based on detection confidence.

---

## Model Performance

### 1. Parkinson's Detection (Audio)
The vocal biomarker system uses a Random Forest Classifier trained on acoustic features extracted via `parselmouth`. It demonstrates exceptional sensitivity for early detection.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **97.61%** | The percentage of total predictions that were correct. |
| **F1 Score** | **96.15%** | Harmonic mean of Precision and Recall (crucial for medical diagnosis). |
| **R2 Score** | **0.86** | Goodness of fit (Statistical measure). |

### 2. Jaundice Detection (Image)
The vision system employs a weighted ensemble of two EfficientNet models to detect diagnostic tinting in skin and sclera.

| Model | Test Accuracy | ROC AUC | Description |
| :--- | :--- | :--- | :--- |
| **Face Model** | **92.00%** | - | Detects general skin discoloration and pallor. |
| **Eye Model** | **86.00%** | **96.43%** | Highly sensitive to scleral icterus (yellowing of eyes). |

---

## Technical Architecture

### Tech Stack
- **Framework**: FastAPI (Python 3.9+)
- **Deep Learning**: PyTorch, EfficientNet (`timm`)
- **Machine Learning**: Scikit-Learn (`sklearn`), Random Forest
- **Audio Processing**: Parselmouth (Praat), Librosa, SoundFile
- **Image Processing**: OpenCV, Pillow
- **Server**: Uvicorn

### Methodology
#### A. Jaundice Detection Algorithm
1. **Preprocessing**: Images are resized and normalized.
2. **Inference**: Parallel execution of Face and Eye EfficientNet models.
3. **Ensemble**: Results are aggregated; if **any** model detects jaundice with high confidence, the system flags a positive result.

#### B. Parkinson's Detection Algorithm
1. **Feature Engineering**: Extracts 16 key acoustic features including **F0 (Pitch)**, **Jitter** (Frequency perturbation), **Shimmer** (Amplitude perturbation), and **HNR** (Harmonic-to-Noise Ratio).
2. **Normalization**: Features are scaled using a pre-fitted `MinMaxScaler`.
3. **Classification**: processed features are fed into the Random Forest model for binary classification.

---

## Installation & Setup

### Prerequisites
- Python 3.9+

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd disease-detection-api
   ```

2. **Install dependencies:**
   ```bash
   # Install generic dependencies
   pip install -r requirements.txt
   
   # Note: Creating a virtual environment is recommended
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Run the server:**
   ```bash
   uvicorn main:app --reload --port 8001
   ```
   The API will be available at `http://0.0.0.0:8001`.

### Docker Setup
```bash
docker build -t disease-api .
docker run -p 8001:8001 disease-api
```

---

## API Endpoints

### Disease Inference (Combined)
- **URL**: `POST /infer`
- **Description**: The primary endpoint. Upload any number of images and/or an audio file. The system automagically routes inputs to the correct models.
- **Body**: `multipart/form-data`
  - `files`: List of image files (Face/Eyes)
  - `audio`: Audio recording (.wav, .mp3)
- **Response**:
  ```json
  {
    "health_score": 40,
    "jaundice_analysis": {
      "detected": true,
      "confidence": 99.5,
      "severity": "Severe",
      "tip": "Significant jaundice signs detected. Seek medical attention promptly."
    },
    "audio_analysis": {
      "pitch_hz": 115.4,
      "jitter_percent": 1.2,
      "parkinsons_detected": true,
      "confidence": 97.6
    },
    "recommendations": [
        "⚠️ Jaundice indicators detected.",
        "⚠️ Audio analysis suggests potential Parkinson's indicators."
    ],
    "images": ["/assets/uuid.jpg"]
  }
  ```

### Health Check
- **URL**: `GET /health`
- **Response**: Returns server status, loaded models, and active device (CPU/MPS/CUDA).

### Demo Inference
- **URL**: `POST /demo/infer`
- **Description**: Returns static demo data for UI testing purposes without running heavy inference.

---