# Disease Detection API

## Overview
The Disease Detection API is a FastAPI-based backend service designed to analyze medical images and provide detailed health assessments. It processes user-uploaded images to detect various health indicators, specifically focusing on facial analysis, pallor detection, and eye condition.

## Purpose
The primary purpose of this server is to act as the inference engine for a health monitoring application. It receives images, processes them (currently using mocked logic for prototyping, designed to be replaced with ML models), and returns structured JSON data containing tailored health advice and analysis.

## Features
- **Multi-Image Processing**: Accepts multiple image uploads in a single request.
- **Detailed Health Analysis**: Returns structured data for:
    - **Face Analysis**: Skin condition, hydration, and fatigue levels.
    - **Pallor Analysis**: Hemoglobin estimates and anemia indicators.
    - **Eye Analysis**: Sclera condition, jaundice signs, and more.
- **Asset Management**: Automatically saves uploaded images to a local `assets/` directory and serves them statically.
- **Scalable Structure**: Built with a modular architecture using Routers, Schemas, and Services.

## Tech Stack
- **Framework**: FastAPI (Python)
- **Server**: Uvicorn
- **Validation**: Pydantic

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Local Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd disease-detection-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

   The server will start at `http://localhost:8000`.

## API Endpoints

### 1. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Returns the operational status of the API.
- **Response**:
  ```json
  {
      "status": "ok"
  }
  ```

### 2. Run Inference
- **URL**: `/infer`
- **Method**: `POST`
- **Description**: Uploads one or more images for health analysis.
- **Body**: `multipart/form-data` with key `files` (List of images).
- **Response**:
  ```json
  {
      "health_score": 85,
      "face_analysis": {
          "skin_condition": "healthy",
          "hydration_level": "95%",
          "fatigue_level": "low",
          "tip": "Keep up the good work!"
      },
      "pallor_analysis": { ... },
      "eye_analysis": { ... },
      "recommendations": [ ... ],
      "images": ["/assets/uuid.jpg", ...]
  }
  ```

## Project Structure
```
disease-detection-api/
├── main.py              # Application entry point
├── routers/             # API route definitions
│   ├── health.py
│   └── inference.py
├── schemas/             # Pydantic models for request/response
├── services/            # Business logic and ML model engine
├── assets/              # Directory for storing uploaded images
├── requirements.txt     # Python dependencies
└── Dockerfile           # Docker configuration
```
