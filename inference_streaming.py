import time
import json
import os
import wave
import pyaudio
import joblib
import librosa
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from queue import Queue
from azure.storage.blob import BlobServiceClient

# Constants for audio streaming
SAMPLING_RATE = 44100  # Sampling rate in Hz
DURATION = 5           # Capture every 2 seconds
CHUNK_SIZE = int(SAMPLING_RATE * DURATION)  # Adjust buffer size

FORMAT = pyaudio.paInt16
CHANNELS = 1

# Queue to store audio chunks
audio_queue = Queue()

# Model artifacts
MODEL_PATH = "model_artifacts/Logistic_Regression_0.2.pkl"
STANDARD_SCALER_PATH = "model_artifacts/scaler_0.2.pkl"
LABEL_ENCODER_PATH = "model_artifacts/label_encoder_0.2.pkl"


class Predictor:
    def __init__(self):
        """Initialize with preloaded model and scalers"""
        self.scaler = joblib.load(STANDARD_SCALER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.model = joblib.load(MODEL_PATH)

    def predict(self, input_data):
        """Make a prediction"""
        input_df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data
        scaled_data = self.scaler.transform(input_df)
        prediction = self.model.predict(scaled_data)
        return self.label_encoder.inverse_transform(prediction)


def spectral_entropy(audio_signal, num_bins=10):
    """Compute spectral entropy."""
    power_spectrum = np.abs(np.fft.fft(audio_signal)) ** 2
    histogram, _ = np.histogram(power_spectrum, bins=num_bins, density=True)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
    return entropy


def envelope_slope(y, sr):
    """Compute attack and decay time."""
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    attack = np.argmax(amplitude_envelope > 0.5 * np.max(amplitude_envelope)) / sr
    decay = (len(y) - np.argmax(amplitude_envelope[::-1] > 0.5 * np.max(amplitude_envelope))) / sr
    return attack, decay


def extract_audio_features(y, sr):
    """Extract audio features from real-time streaming data."""
    features = {}

    # 1. Zero Crossing Rate (ZCR)
    features["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y))

    # 2. Spectral Centroid
    features["spectral_centroid_mean"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # 3. Spectral Rolloff
    features["spectral_rolloff_mean"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # 4. Spectral Bandwidth
    features["spectral_bandwidth_mean"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # 5. Root Mean Square (RMS) Energy
    features["rms_mean"] = np.mean(librosa.feature.rms(y=y))

    # 6. Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])

    # 7. Chroma Features (STFT)
    features["chroma_mean"] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

    # 8. Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast_mean"] = np.mean(spectral_contrast)

    # 9. Tonnetz (Tonal Centroid Features)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features["tonnetz_mean"] = np.mean(tonnetz)

    # 10. Onset Strength
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    features["onset_strength_mean"] = np.mean(onset_strength)

    # 11. Harmonic and Percussive RMS
    harmonic, percussive = librosa.effects.hpss(y)
    features["harmonic_rms"] = np.mean(librosa.feature.rms(y=harmonic))
    features["percussive_rms"] = np.mean(librosa.feature.rms(y=percussive))

    # 12. Chromagram from Constant-Q Transform (CQT)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    features["chroma_cqt_mean"] = np.mean(chroma_cqt)

    # 13. Spectral Flatness
    features["spectral_flatness_mean"] = np.mean(librosa.feature.spectral_flatness(y=y))

    # 14. Spectral Entropy
    features["spectral_entropy"] = spectral_entropy(y)

    # 15. Crest Factor
    features["crest_factor"] = np.max(np.abs(y)) / features["rms_mean"]

    # 16. Attack Time and Decay Time
    attack_time, decay_time = envelope_slope(y, sr)
    features["attack_time"] = attack_time
    features["decay_time"] = decay_time

    # 17. Skewness and Kurtosis
    features["skewness"] = skew(y)
    features["kurtosis"] = kurtosis(y)

    return features


def audio_callback(in_data, frame_count, time_info, status):
    """Callback function for PyAudio stream."""
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    audio_queue.put(audio_data)
    return (in_data, pyaudio.paContinue)


# Initialize PyAudio
p = pyaudio.PyAudio()

# Open an audio stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNK_SIZE, stream_callback=audio_callback)

print("Starting real-time audio feature extraction every 2 seconds...")
stream.start_stream()

# Process the real-time audio stream every 2 seconds
try:
    predictor = Predictor()
    while True:
        time.sleep(DURATION)  # Wait for 2 seconds
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if not audio_queue.empty():
            audio_chunk = audio_queue.get()
            features = extract_audio_features(audio_chunk, SAMPLING_RATE)

            # Predict using the model
            start_time = time.time()
            predictions = predictor.predict(features)
            end_time = time.time()
            print(f"Prediction took {end_time - start_time:.2f} seconds")

            # Prepare result
            result = features | {
                "timestamp": timestamp,
                "prediction": predictions[0],
            }

            print(f"Predicted result: {result}")

except KeyboardInterrupt:
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
