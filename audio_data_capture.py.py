import time
import json
import csv
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

    start_time = time.time()
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    features["mean_pitch"] = np.nan_to_num(np.nanmean(f0), nan=0.0)
    pitch_time = time.time() - start_time
    print(f"Pitch calculation took {pitch_time:.4f} seconds")


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


def save_to_json(data, filename):
    """
    Save data to a JSON file.
    """
    start_time = time.time()
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    end_time = time.time()
    print(f"Saving to JSON took {end_time - start_time:.2f} seconds")


# This runs once to initialize the CSV header
def initialize_csv_if_needed(csv_file, feature_length):
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['filename'] + [f'feature_{i+1}' for i in range(feature_length)]
            writer.writerow(header)

# Append features for one audio file
def append_features_to_csv(csv_file, filename, features):
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([filename] + list(features))


if __name__ == "__main__":

    state="off"
    file_path="<write-your-path-here>" # Changed for demo purposes

    # Create the directory if it doesn't exist
    if not os.path.exists(file_path):
        try:
            os.makedirs(file_path)  # Create the main directory
            os.makedirs(os.path.join(file_path, state))  # Create the subdirectory for "off" state

        except OSError as e:
            print(f"Error creating directory: {e}")
            exit()  # Exit the script if directory creation fails
    else:
        print(f"Directory {file_path} already exists.")


    while True:

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        audio_filename = f"{file_path}/{state}/Device1_{timestamp}_{state}.wav"

        print(f"Recording audio to {audio_filename}")
        try:
            record_audio(audio_filename)
            print(f"Audio saved to {audio_filename}")
        except Exception as e:
            print(f"Error recording audio: {e}")
            time.sleep(1)
            continue #skip to next iteration if there's an issue with recording

        # Extract features
        start_time = time.time()
        print(f"Extracting features from {audio_filename}")
        try:
            features = extract_audio_features_basic_profile(audio_filename)

        except Exception as e:
            print(f"Error extracting features: {e}")
            time.sleep(1)
            continue #skip to next iteration if there's an issue with extracting features


        end_time = time.time()
        print(f"Features extraction basic took {end_time - start_time:.2f} seconds")

        time.sleep(1)