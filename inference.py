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
MODEL_PATH = "model_artifacts/decision_tree.joblib"
STANDARD_SCALER_PATH = "model_artifacts/standard_scaler.joblib"
LABEL_ENCODER_PATH = "model_artifacts/label_encoder.joblib"

class Predictor:
    def __init__(self, model_name=None):
        """Initialize with specific model or best model"""
        self.scaler = joblib.load('preprocessing/scaler.pkl')
        self.label_encoder = joblib.load('preprocessing/label_encoder.pkl')

        if model_name:
            self.model = joblib.load(f'models/{model_name.replace(" ", "_")}.pkl')
        else:
            # Automatically select best model by F1 score
            metrics = pd.read_csv('model_metrics.csv')
            best_model = metrics.loc[metrics['F1'].idxmax()]['Model']
            self.model = joblib.load(f'models/{best_model.replace(" ", "_")}.pkl')

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


def extract_audio_features_basic(audio_file):
    """
    Extract audio features from the input audio file.

    Parameters:
        audio_file (str): Path to the audio file.

    Returns:
        dict: Dictionary of extracted audio features.
    """
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Feature extraction
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

    # 12. Pitch (Fundamental Frequency)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    features["mean_pitch"] = np.nan_to_num(np.nanmean(f0), nan=0.0)

    # 13. Tempo (BPM)
    #tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)[0]  # Updated for newer versions of librosa
    #features["tempo"] = tempo

    # 14. Chromagram from Constant-Q Transform (CQT)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    features["chroma_cqt_mean"] = np.mean(chroma_cqt)

    # 15. Spectral Flatness
    features["spectral_flatness_mean"] = np.mean(librosa.feature.spectral_flatness(y=y))

    # 16. Spectral Entropy
    def spectral_entropy(audio_signal, num_bins=10):
        power_spectrum = np.abs(np.fft.fft(audio_signal)) ** 2
        histogram, _ = np.histogram(power_spectrum, bins=num_bins, density=True)
        entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
        return entropy

    features["spectral_entropy"] = spectral_entropy(y)

    # 17. Crest Factor
    features["crest_factor"] = np.max(np.abs(y)) / features["rms_mean"]

    # 18. Attack Time and Decay Time
    def envelope_slope(y, sr):
        analytic_signal = hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)
        attack = np.argmax(amplitude_envelope > 0.5 * np.max(amplitude_envelope)) / sr
        decay = (len(y) - np.argmax(amplitude_envelope[::-1] > 0.5 * np.max(amplitude_envelope))) / sr
        return attack, decay

    attack_time, decay_time = envelope_slope(y, sr)
    features["attack_time"] = attack_time
    features["decay_time"] = decay_time

    # 19. Skewness and Kurtosis
    features["skewness"] = skew(y)
    features["kurtosis"] = kurtosis(y)

    return features


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


if __name__ == "__main__":
    predictor = Predictor()

    while True:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        audio_filename = f"Device1_{timestamp}.wav"
        json_filename = f"Device1_{timestamp}.json"

        print(f"Recording audio to {audio_filename}")
        # Record audio
        record_audio(audio_filename)
        print(f"Audio saved to {audio_filename}")
        
        # Extract features
        print(f"Extracting features from {audio_filename}")
        features = extract_audio_features_basic(audio_filename)
        print(f"Features extracted")

        # Predict using the model
        start_time = time.time()
        predictions = predictor.predict(features)
        end_time = time.time()
        print(f"Prediction took {end_time - start_time:.2f} seconds")

        print(f"Prediction: {predictions[0]}")

        # Prepare data as a single dictionary
        result = features | {
            "timestamp": timestamp,
            "prediction": predictions[0],
            "audio_filename": audio_filename  # Store the single prediction value
        } 

        print("production result: {result}")

        time.sleep(2)  # Wait 5 seconds before the next recording
