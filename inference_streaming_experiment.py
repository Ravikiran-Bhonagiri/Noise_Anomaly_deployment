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


def extract_audio_features_basic_profile(audio_file):
    """
    Extract audio features from the input audio file.

    Parameters:
        audio_file (str): Path to the audio file.

    Returns:
        dict: Dictionary of extracted audio features.
    """
    # Load the audio file
    start_time = time.time()
    y, sr = librosa.load(audio_file, sr=None)
    load_time = time.time() - start_time
    print(f"Audio loading took {load_time:.4f} seconds")

    # Feature extraction
    features = {}

    # 1. Zero Crossing Rate (ZCR)
    start_time = time.time()
    features["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y))
    zcr_time = time.time() - start_time
    print(f"Zero Crossing Rate calculation took {zcr_time:.4f} seconds")

    # 2. Spectral Centroid
    start_time = time.time()
    features["spectral_centroid_mean"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_time = time.time() - start_time
    print(f"Spectral Centroid calculation took {spectral_centroid_time:.4f} seconds")

    # 3. Spectral Rolloff
    start_time = time.time()
    features["spectral_rolloff_mean"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_rolloff_time = time.time() - start_time
    print(f"Spectral Rolloff calculation took {spectral_rolloff_time:.4f} seconds")

    # 4. Spectral Bandwidth
    start_time = time.time()
    features["spectral_bandwidth_mean"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_time = time.time() - start_time
    print(f"Spectral Bandwidth calculation took {spectral_bandwidth_time:.4f} seconds")

    # 5. Root Mean Square (RMS) Energy
    start_time = time.time()
    features["rms_mean"] = np.mean(librosa.feature.rms(y=y))
    rms_time = time.time() - start_time
    print(f"RMS Energy calculation took {rms_time:.4f} seconds")

    # 6. Mel-Frequency Cepstral Coefficients (MFCCs)
    start_time = time.time()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])
    mfcc_time = time.time() - start_time
    print(f"MFCC calculation took {mfcc_time:.4f} seconds")

    # 7. Chroma Features (STFT)
    start_time = time.time()
    features["chroma_mean"] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_time = time.time() - start_time
    print(f"Chroma Features calculation took {chroma_time:.4f} seconds")

    # 8. Spectral Contrast
    start_time = time.time()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast_mean"] = np.mean(spectral_contrast)
    spectral_contrast_time = time.time() - start_time
    print(f"Spectral Contrast calculation took {spectral_contrast_time:.4f} seconds")

    # 9. Tonnetz (Tonal Centroid Features)
    start_time = time.time()
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features["tonnetz_mean"] = np.mean(tonnetz)
    tonnetz_time = time.time() - start_time
    print(f"Tonnetz calculation took {tonnetz_time:.4f} seconds")

    # 10. Onset Strength
    start_time = time.time()
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    features["onset_strength_mean"] = np.mean(onset_strength)
    onset_strength_time = time.time() - start_time
    print(f"Onset Strength calculation took {onset_strength_time:.4f} seconds")

    # 11. Harmonic and Percussive RMS
    start_time = time.time()
    harmonic, percussive = librosa.effects.hpss(y)
    features["harmonic_rms"] = np.mean(librosa.feature.rms(y=harmonic))
    features["percussive_rms"] = np.mean(librosa.feature.rms(y=percussive))
    hpss_time = time.time() - start_time
    print(f"Harmonic and Percussive RMS calculation took {hpss_time:.4f} seconds")

    # 12. Pitch (Fundamental Frequency)
    '''
    start_time = time.time()
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    features["mean_pitch"] = np.nan_to_num(np.nanmean(f0), nan=0.0)
    pitch_time = time.time() - start_time
    print(f"Pitch calculation took {pitch_time:.4f} seconds")
    '''

    # 13. Chromagram from Constant-Q Transform (CQT)
    start_time = time.time()
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    features["chroma_cqt_mean"] = np.mean(chroma_cqt)
    chroma_cqt_time = time.time() - start_time
    print(f"Chromagram from CQT calculation took {chroma_cqt_time:.4f} seconds")

    # 14. Spectral Flatness
    start_time = time.time()
    features["spectral_flatness_mean"] = np.mean(librosa.feature.spectral_flatness(y=y))
    spectral_flatness_time = time.time() - start_time
    print(f"Spectral Flatness calculation took {spectral_flatness_time:.4f} seconds")

    # 16. Spectral Entropy
    start_time = time.time()
    def spectral_entropy(audio_signal, num_bins=10):
        power_spectrum = np.abs(np.fft.fft(audio_signal)) ** 2
        histogram, _ = np.histogram(power_spectrum, bins=num_bins, density=True)
        entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
        return entropy
    features["spectral_entropy"] = spectral_entropy(y)
    spectral_entropy_time = time.time() - start_time
    print(f"Spectral Entropy calculation took {spectral_entropy_time:.4f} seconds")
    
    # 17. Crest Factor
    start_time = time.time()
    features["crest_factor"] = np.max(np.abs(y)) / features["rms_mean"]
    crest_factor_time = time.time() - start_time
    print(f"Crest Factor calculation took {crest_factor_time:.4f} seconds")
    
    # 18. Attack Time and Decay Time
    start_time = time.time()
    def envelope_slope(y, sr):
        analytic_signal = hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)
        attack = np.argmax(amplitude_envelope > 0.5 * np.max(amplitude_envelope)) / sr
        decay = (len(y) - np.argmax(amplitude_envelope[::-1] > 0.5 * np.max(amplitude_envelope))) / sr
        return attack, decay
    attack_time, decay_time = envelope_slope(y, sr)
    features["attack_time"] = attack_time
    features["decay_time"] = decay_time
    envelope_slope_time = time.time() - start_time
    print(f"Attack and Decay Time calculation took {envelope_slope_time:.4f} seconds")

    # 19. Skewness and Kurtosis
    start_time = time.time()
    features["skewness"] = skew(y)
    features["kurtosis"] = kurtosis(y)
    skew_kurtosis_time = time.time() - start_time
    print(f"Skewness and Kurtosis calculation took {skew_kurtosis_time:.4f} seconds")
    return features

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
        time.sleep(1)  # Wait for 2 seconds
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

            print(f"Predicted streaming result: {result}")

            # --- Save Audio Chunk as a WAV File ---
            wav_filename = f"audio_chunk_{timestamp}.wav"
            with wave.open(wav_filename, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(SAMPLING_RATE)
                wf.writeframes(audio_chunk.tobytes())  # Convert to byte format
            print(f"Saved audio: {wav_filename}")

            # Extract features
            start_time = time.time()
            print(f"Extracting features from {wav_filename}")
            features = extract_audio_features_basic_profile(wav_filename)
            end_time = time.time()
            print(f"Features extraction basic took {end_time - start_time:.2f} seconds")

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
                "audio_filename": wav_filename  # Store the single prediction value
            }

            print(f"production basic result: {result1}")

            #os.remove(audio_filename)

except KeyboardInterrupt:
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()