import time
import json
import os
import time
import wave
import json
import pyaudio
import pandas as pd
import joblib
import librosa
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from azure.storage.blob import BlobServiceClient

# Constants for audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

## model artifacts
MODEL_PATH = "model_artifacts/Logistic_Regression_0.2.pkl"
STANDARD_SCALER_PATH = "model_artifacts/scaler_0.2.pkl"
LABEL_ENCODER_PATH = "model_artifacts/label_encoder_0.2.pkl"


class Predictor:
    def __init__(self, model_name=None):
        """Initialize with specific model or best model"""
        self.scaler = joblib.load(STANDARD_SCALER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.model = joblib.load(MODEL_PATH)

    def predict(self, input_data):
        """Make prediction on new data"""
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data

        # Preprocess input
        scaled_data = self.scaler.transform(input_df)

        # Make prediction
        prediction = self.model.predict(scaled_data)
        return self.label_encoder.inverse_transform(prediction)





def record_audio(filename):
    """
    Record audio and save to a file.
    """
    start_time = time.time()
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    end_time = time.time()
    print(f"Audio recording took {end_time - start_time:.2f} seconds")

def upload_to_blob_storage(local_file_path, blob_name):
    """
    Upload a file to Azure Blob Storage.
    """
    start_time = time.time()
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    with open(local_file_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data)
    os.remove(local_file_path)
    end_time = time.time()
    print(f"Uploading to blob storage took {end_time - start_time:.2f} seconds")

def save_to_json(data, filename):
    """
    Save data to a JSON file.
    """
    start_time = time.time()
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    end_time = time.time()
    print(f"Saving to JSON took {end_time - start_time:.2f} seconds")


import numpy as np
import pyaudio
import librosa
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
import queue

# Audio stream parameters
SAMPLING_RATE = 22050  # Common sampling rate for Librosa
CHUNK_SIZE = 1024      # Buffer size
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Queue to store audio chunks
audio_queue = queue.Queue()

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
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
    audio_queue.put(audio_data)
    return (in_data, pyaudio.paContinue)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open an audio stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLING_RATE,
                input=True, frames_per_buffer=CHUNK_SIZE, stream_callback=audio_callback)

print("Starting real-time audio feature extraction...")
stream.start_stream()

# Process the real-time audio stream
try:
    while True:
        
        predictor = Predictor()
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if not audio_queue.empty():
            audio_chunk = audio_queue.get()
            features = extract_audio_features(audio_chunk, SAMPLING_RATE)
            # Predict using the model
            start_time = time.time()
            predictions = predictor.predict(features)
            end_time = time.time()
            print(f"Prediction took {end_time - start_time:.2f} seconds")

            print(f"Prediction: {predictions[0]}")

            # Prepare data as a single dictionary
            result1 = features | {
                "timestamp": timestamp,
                "prediction": predictions[0],
            }

            print(f"production basic result: {result1}")


            time.sleep(1)  # Wait 5 seconds before the next recording
except KeyboardInterrupt:
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()