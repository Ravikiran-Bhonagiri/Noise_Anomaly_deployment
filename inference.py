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
RECORD_SECONDS = 1

## model artifacts
MODEL_PATH = "model_artifacts_xgboost/XGBoost_test_size_0.2.pkl"
STANDARD_SCALER_PATH = "model_artifacts_xgboost/scaler.pkl"
LABEL_ENCODER_PATH = "model_artifacts_xgboost/label_encoder.pkl"

def compare_dictionaries(dict1, dict2, float_tol=1e-9):
    """
    Compare two dictionaries for key/value discrepancies with numerical tolerance for floats
    
    Parameters:
        dict1 (dict): First dictionary to compare
        dict2 (dict): Second dictionary to compare
        float_tol (float): Absolute tolerance for float comparisons (default: 1e-9)
        
    Returns:
        dict: Dictionary containing:
            - missing_keys: Keys present in one dict but not the other
            - type_mismatches: Keys with different value types
            - value_mismatches: Keys with differing values
    """
    report = {
        'missing_keys': {
            'in_dict1': [],
            'in_dict2': []
        },
        'type_mismatches': {},
        'value_mismatches': {}
    }
    
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in all_keys:
        # Check for missing keys
        if key not in dict1:
            report['missing_keys']['in_dict1'].append(key)
            continue
        if key not in dict2:
            report['missing_keys']['in_dict2'].append(key)
            continue
            
        val1 = dict1[key]
        val2 = dict2[key]
        
        # Check type mismatches
        if type(val1) != type(val2):
            report['type_mismatches'][key] = f"{type(val1)} vs {type(val2)}"
            continue
            
        # Value comparison logic
        try:
            if isinstance(val1, float):
                # Float comparison with tolerance
                if not np.isclose(val1, val2, atol=float_tol):
                    report['value_mismatches'][key] = {
                        'dict1_value': val1,
                        'dict2_value': val2,
                        'absolute_diff': abs(val1 - val2)
                    }
            elif isinstance(val1, np.ndarray):
                # Numpy array comparison
                if not np.array_equal(val1, val2, equal_nan=True):
                    report['value_mismatches'][key] = "Array values differ"
            else:
                # Standard comparison for other types
                if val1 != val2:
                    report['value_mismatches'][key] = {
                        'dict1_value': val1,
                        'dict2_value': val2
                    }
        except Exception as e:
            report['value_mismatches'][key] = f"Comparison failed: {str(e)}"
    
    return report

def format_comparison_report(report):
    """Format the comparison report into a human-readable string"""
    output = []
    
    # Missing keys
    if report['missing_keys']['in_dict1']:
        output.append("Keys missing in first dictionary:")
        output.extend(f" - {key}" for key in report['missing_keys']['in_dict1'])
        
    if report['missing_keys']['in_dict2']:
        output.append("\nKeys missing in second dictionary:")
        output.extend(f" - {key}" for key in report['missing_keys']['in_dict2'])
    
    # Type mismatches
    if report['type_mismatches']:
        output.append("\nType mismatches:")
        output.extend(f" - {key}: {types}" for key, types in report['type_mismatches'].items())
    
    # Value mismatches
    if report['value_mismatches']:
        output.append("\nValue mismatches:")
        for key, details in report['value_mismatches'].items():
            if isinstance(details, dict):
                line = f" - {key}:"
                line += f"\n   First value:  {details['dict1_value']}"
                line += f"\n   Second value: {details['dict2_value']}"
                if 'absolute_diff' in details:
                    line += f"\n   Absolute difference: {details['absolute_diff']:.2e}"
                output.append(line)
            else:
                output.append(f" - {key}: {details}")
    
    return "\n".join(output) if output else "Dictionaries are identical"


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


import librosa
import numpy as np
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis

def extract_audio_features_old(audio_file):
    """
    Extract audio features from the input audio file.
    """
    start_time = time.time()
    y, sr = librosa.load(audio_file, sr=None)
    features = {}

    # Audio feature extraction
    features["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y))
    features["spectral_centroid_mean"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["spectral_rolloff_mean"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features["spectral_bandwidth_mean"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features["rms_mean"] = np.mean(librosa.feature.rms(y=y))

    # MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])

    features["chroma_mean"] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    features["spectral_contrast_mean"] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    features["tonnetz_mean"] = np.mean(librosa.feature.tonnetz(y=y, sr=sr))
    features["onset_strength_mean"] = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    # Harmonic and percussive RMS
    
    harmonic, percussive = librosa.effects.hpss(y)
    features["harmonic_rms"] = np.mean(librosa.feature.rms(y=harmonic))
    features["percussive_rms"] = np.mean(librosa.feature.rms(y=percussive))
    
    # Pitch and tempo
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    features["mean_pitch"] = np.mean(pitches[pitches > 0])
    #features["tempo"] = librosa.beat.tempo(y=y, sr=sr)[0]
    # Chroma CQT and spectral flatness
    features["chroma_cqt_mean"] = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr))
    features["spectral_flatness_mean"] = np.mean(librosa.feature.spectral_flatness(y=y))
    # Spectral entropy
    spectral_energy = y ** 2
    spectral_probs = spectral_energy / np.sum(spectral_energy + 1e-7)
    features["spectral_entropy"] = -np.sum(spectral_probs * np.log(spectral_probs + 1e-7))
    # Crest factor
    features["crest_factor"] = np.max(np.abs(y)) / np.mean(np.abs(y))
    # Attack and decay time (based on onset envelope)
    envelope = librosa.onset.onset_strength(y=y, sr=sr)
    features["attack_time"] = np.argmax(envelope) / sr
    features["decay_time"] = (len(envelope) - np.argmax(envelope[::-1])) / sr
    # Skewness and kurtosis
    features["skewness"] = skew(y)
    features["kurtosis"] = kurtosis(y)

    end_time = time.time()
    print(f"Feature extraction took {end_time - start_time:.2f} seconds")
    return features

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

def extract_audio_features_optimized(audio_file):
    """
    Extract audio features with optimized implementation
    
    Parameters:
        audio_file (str): Path to the audio file
        
    Returns:
        dict: Dictionary of extracted audio features
    """
    # Load audio with resampling prevention
    y, sr = librosa.load(audio_file, sr=None)
    
    features = {}
    
    # Helper functions
    def _get_mean(feature_func, *args, **kwargs):
        return np.nanmean(feature_func(*args, **kwargs))
    
    def _spectral_entropy(signal, bins=10):
        psd = np.abs(np.fft.fft(signal)) ** 2
        hist = np.histogram(psd, bins=bins, density=True)[0]
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    # Temporal features
    features.update({
        "zcr_mean": _get_mean(librosa.feature.zero_crossing_rate, y),
        "rms_mean": _get_mean(librosa.feature.rms, y=y)
    })

    features["crest_factor"] = np.max(np.abs(y)) / (features["rms_mean"] + 1e-7)
    
    # Spectral features
    spectral_features = {
        'centroid': librosa.feature.spectral_centroid,
        'rolloff': librosa.feature.spectral_rolloff,
        'bandwidth': librosa.feature.spectral_bandwidth,
        'contrast': librosa.feature.spectral_contrast,
        'flatness': librosa.feature.spectral_flatness
    }
    
    for name, func in spectral_features.items():
        key = f"spectral_{name}_mean"
        features[key] = _get_mean(func, y=y, sr=sr)
    
    # Chroma features
    chroma_types = {
        'stft': librosa.feature.chroma_stft,
        'cqt': librosa.feature.chroma_cqt
    }
    
    for chroma_type, func in chroma_types.items():
        features[f"chroma_{chroma_type}_mean"] = _get_mean(func, y=y, sr=sr)
    
    # MFCCs (vectorized computation)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.update({
        f"mfcc_{i+1}_mean": np.mean(mfccs[i]) 
        for i in range(13)
    })
    
    # Harmonic analysis
    harmonic, percussive = librosa.effects.hpss(y)
    features.update({
        "harmonic_rms": _get_mean(librosa.feature.rms, y=harmonic),
        "percussive_rms": _get_mean(librosa.feature.rms, y=percussive),
        "tonnetz_mean": _get_mean(librosa.feature.tonnetz, y=harmonic, sr=sr)
    })
    
    # Pitch detection with PYIN
    f0, _, _ = librosa.pyin(y, 
                          fmin=librosa.note_to_hz('C2'), 
                          fmax=librosa.note_to_hz('C7'))
    features["mean_pitch"] = np.nan_to_num(np.nanmean(f0), nan=0.0)
    
    # Temporal characteristics
    envelope = np.abs(hilbert(y))
    threshold = 0.5 * envelope.max()
    attack_decay = {
        "attack_time": np.argmax(envelope > threshold) / sr,
        "decay_time": (len(y) - np.argmax(envelope[::-1] > threshold)) / sr
    }
    features.update(attack_decay)
    
    # Statistical features
    stats = {
        "skewness": skew(y),
        "kurtosis": kurtosis(y),
        "spectral_entropy": _spectral_entropy(y),
        "onset_strength_mean": _get_mean(librosa.onset.onset_strength, y=y, sr=sr)
    }
    features.update(stats)
    
    return features

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
        start_time = time.time()
        print(f"Extracting features from {audio_filename}")
        features = extract_audio_features_basic_profile(audio_filename)
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
            "audio_filename": audio_filename  # Store the single prediction value
        }

        print(f"production basic result: {result1}")

        os.remove(audio_filename)
        
        time.sleep(1)
