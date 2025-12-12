# Parkinson’s Disease Detection using Vocal Biomarkers

## 1\. Project Overview

This project implements a Machine Learning system to detect Parkinson's Disease (PD) based on acoustic analysis of voice recordings. Parkinson's disease affects the motor control of the vocal folds, resulting in specific vocal degradations (dysphonia). This system extracts these "vocal biomarkers" in real-time and classifies the subject as Healthy or having Parkinson's.

## 2\. System Architecture

The system consists of two main components:

1.  **Training Module :** Preprocesses data, handles class imbalance, and trains a Random Forest Classifier.
2.  **Real-Time Inference Module :** Captures live audio, extracts acoustic features using Praat, and generates a prediction.

## 3\. Methodology & Algorithms

### A. Feature Engineering (Acoustic Analysis)

The model relies on **16 specific acoustic features** calculated from the voice signal. These are extracted using the `parselmouth` library (a Python interface for Praat).

  * **Fundamental Frequency (F0):** Average, Maximum, and Minimum pitch.
  * **Jitter (Frequency Perturbation):** Measures cycle-to-cycle fluctuations in pitch. High jitter indicates a lack of control over vocal fold vibration.
  * **Shimmer (Amplitude Perturbation):** Measures cycle-to-cycle fluctuations in loudness.
  * **Harmonic-to-Noise Ratio (HNR):** Measures the ratio of periodic sound (voice) to aperiodic sound (noise/breathiness).

### B. Data Preprocessing 

To ensure robust model performance, the following preprocessing steps are applied:

1.  **Feature Selection:** The dataset is filtered to include only the 16 features calculable in real-time.
2.  **Handling Class Imbalance (SMOTE):**
      * Medical datasets often have fewer "Positive" (Parkinson's) cases than "Negative" (Healthy) cases.
      * **SMOTE (Synthetic Minority Over-sampling Technique)** is used to generate synthetic examples of the minority class, ensuring the model doesn't become biased toward the majority class.
3.  **Data Scaling:**
      * A `MinMaxScaler` is applied to normalize all features to a range of `[-1, 1]`. This ensures that features with larger raw numbers (like Hz) don't dominate features with small percentages.

### C. Model Classification

  * **Algorithm:** Random Forest Classifier.
  * **Why Random Forest?** It is an ensemble learning method that constructs a multitude of decision trees. It handles non-linear relationships well and is highly resistant to overfitting compared to single decision trees.

-----

## 4\. Implementation Details

### Phase 1: Model Training 

This script is responsible for building the prediction engine.

1.  **Ingestion:** Loads the `parkinsons.data` CSV.
2.  **Resampling:** Applies SMOTE to balance the dataset.
3.  **Splitting:** Divides data into 80% Training and 20% Testing sets.
4.  **Scaling:** Fits the `MinMaxScaler` on training data.
5.  **Training:** Fits the Random Forest model.
6.  **Serialization:** Saves the trained model (`rf_model_lite.pkl`) and the scaler (`scaler_lite.pkl`) for later use.

### Phase 2: Live Detection 

This script allows for live testing via a microphone.

1.  **Audio Capture:** Uses `sounddevice` to record 10 seconds of audio.
2.  **Signal Processing:**
      * The audio is analyzed using `parselmouth` (Praat).
      * Specific metrics (Jitter, Shimmer, RAP, PPQ, etc.) are calculated mathematically to match the training data format.
      * *Note:* NaNs (Not a Number) are handled by replacing them with 0 to prevent crashes during silence.
3.  **Inference:**
      * The features are scaled using the saved `scaler_lite.pkl`.
      * The `rf_model_lite.pkl` predicts the class (0 or 1).
      * Prediction probabilities are calculated to provide a "Confidence Score."

-----

## 5\. Performance Metrics

The Random Forest Classifier was evaluated on the test set and yielded the following performance metrics, outperforming other tested classifiers:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **97.61%** | The percentage of total predictions that were correct. |
| **F1 Score** | **96.15%** | The harmonic mean of Precision and Recall (crucial for medical diagnosis). |
| **R2 Score** | **0.86** | Goodness of fit (Statistical measure). |

-----

## 6\. How to Run

### Prerequisites

Install the required libraries:

```bash
pip install pandas joblib scikit-learn imbalanced-learn sounddevice scipy parselmouth numpy
```

*Note: You may need to install PortAudio drivers separately depending on your OS (e.g., `sudo apt-get install libportaudio2` on Linux).*

