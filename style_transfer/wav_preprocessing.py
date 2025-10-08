# /HybridAI-Music-Composer/style_transfer/wav_preprocessing.py

import librosa
import numpy as np

# Constants
DEFAULT_FS = 22050
SEGMENT_DURATION = 0.25  # seconds

def load_and_segment_audio(wav_path, segment_duration=SEGMENT_DURATION, sr=DEFAULT_FS):
    """
    Load audio, resample, split into segments, and normalize each segment.
    """
    try:
        y, sr_loaded = librosa.load(wav_path, sr=sr, mono=True)
    except Exception as e:
        print(f"ERROR loading WAV {wav_path}: {e}")
        return np.array([], dtype=np.float32)
        
    samples_per_segment = int(segment_duration * sr)
    num_segments = len(y) // samples_per_segment

    segments = []
    for i in range(num_segments):
        segment = y[i * samples_per_segment: (i + 1) * samples_per_segment]
        # Robust normalization: normalize each segment before passing to feature extraction
        max_abs = np.max(np.abs(segment)) + 1e-9
        normalized = segment / max_abs
        segments.append(normalized.astype(np.float32))

    return np.array(segments, dtype=np.float32)

def extract_features(segment, sr=DEFAULT_FS, n_mfcc=13):
    """
    Extract audio features from a segment (f0, spectral centroid, rolloff, MFCCs).
    """
    # 1. Pitch estimation (f0)
    f0, _, _ = librosa.pyin(segment,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'),
                            sr=sr)
    # Use median instead of mean for robustness against outliers
    f0_mean = np.nanmedian(f0) if np.any(np.isfinite(f0)) else 0.0

    # 2. Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, roll_percent=0.85).mean()

    # 3. MFCC
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc).mean(axis=1)

    feature_vector = np.hstack([f0_mean, spectral_centroid, spectral_rolloff, mfccs]).astype(np.float32)
    return feature_vector

def create_feature_dataset(wav_path, segment_duration=SEGMENT_DURATION, sr=DEFAULT_FS, n_mfcc=13):
    """
    Processes wav file into feature and segment datasets for training.
    """
    segments = load_and_segment_audio(wav_path, segment_duration, sr)
    if segments.size == 0:
         return np.array([[]]), np.array([[]])
         
    features = np.array([extract_features(seg, sr, n_mfcc) for seg in segments], dtype=np.float32)
    return features, segments
