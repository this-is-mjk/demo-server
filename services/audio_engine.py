import parselmouth
from parselmouth.praat import call
import joblib
import numpy as np
import os

class AudioEngine:
    def __init__(self, model_path="rf_model_lite.pkl", scaler_path="scaler_lite.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.clf = None
        self.scaler = None
        self.load_models()

    def load_models(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.clf = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("Audio models loaded successfully.")
            else:
                print(f"Warning: Model files not found ({self.model_path}, {self.scaler_path}). Audio analysis will be limited.")
        except Exception as e:
            print(f"Error loading audio models: {e}")

    def extract_features(self, audio_path):
        """
        Extracts the 16 jitter/shimmer/pitch features using Praat.
        """
        try:
            sound = parselmouth.Sound(audio_path)
            
            # 1. Pitch Analysis
            pitch = sound.to_pitch()
            f0_mean = call(pitch, "Get mean", 0, 0, "Hertz") # MDVP:Fo(Hz)
            f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic") # MDVP:Fhi(Hz)
            f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic") # MDVP:Flo(Hz)

            # 2. Pulse Analysis (for Jitter/Shimmer)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
            
            # Jitter
            jitter_percent = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) # MDVP:Jitter(%)
            jitter_abs     = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) # MDVP:Jitter(Abs)
            jitter_rap     = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) # MDVP:RAP
            jitter_ppq5    = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) # MDVP:PPQ
            jitter_ddp     = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) # Jitter:DDP

            # Shimmer
            shimmer_local  = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) # MDVP:Shimmer
            shimmer_db     = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6) # MDVP:Shimmer(dB)
            shimmer_apq3   = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) # Shimmer:APQ3
            shimmer_apq5   = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6) # Shimmer:APQ5
            shimmer_apq11  = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6) # MDVP:APQ
            shimmer_dda    = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6) # Shimmer:DDA

            # 3. Harmonicity
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0) # HNR
            
            # Simple NHR calculation (Noise-to-Harmonics Ratio)
            if hnr == 0 or np.isnan(hnr):
                nhr = 0
                hnr = 0 
            else:
                nhr = 1 / hnr

            features = [
                f0_mean, f0_max, f0_min,
                jitter_percent * 100, jitter_abs, jitter_rap * 100, jitter_ppq5 * 100, jitter_ddp * 100, 
                shimmer_local * 100, shimmer_db, shimmer_apq3 * 100, shimmer_apq5 * 100, shimmer_apq11 * 100, shimmer_dda * 100,
                nhr, hnr
            ]
            
            features = [0 if np.isnan(x) else x for x in features]
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def analyze_audio(self, audio_path):
        """
        Analyzes audio file for Parkinson's detection.
        Returns dictionary with analysis results.
        """
        if not self.clf or not self.scaler:
             return {"error": "Model not loaded"}
             
        features = self.extract_features(audio_path)
        
        if features is None:
            return {"error": "Feature extraction failed"}

        if features[0][0] == 0:
             return {"warning": "No voice detected"}

        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.clf.predict(features_scaled)
            probability_positive = self.clf.predict_proba(features_scaled)[0][1]
            
            is_parkinsons = bool(prediction[0] == 1)
            confidence = probability_positive if is_parkinsons else 1 - probability_positive

            return {
                "pitch_hz": round(features[0][0], 2),
                "jitter_percent": round(features[0][3], 4),
                "shimmer_percent": round(features[0][8], 4),
                "parkinsons_detected": is_parkinsons,
                "confidence": round(confidence * 100, 2)
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

audio_engine = AudioEngine()
