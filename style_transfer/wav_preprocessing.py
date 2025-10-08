
import librosa
import numpy as np

# Constants
DEFAULT_FS = 22050
SEGMENT_DURATION = 0.25  # seconds

def load_and_segment_audio(wav_path, segment_duration=SEGMENT_DURATION, sr=DEFAULT_FS):
    """
    Load audio from wav_path, resample to sr, split into segments of segment_duration seconds,
    normalize each segment.

    Args:
        wav_path (str): Path to the WAV file.
        segment_duration (float): Seconds per segment.
        sr (int): Sampling rate.

    Returns:
        np.ndarray: Array of shape (num_segments, samples_per_segment) with normalized audio segments.
    """
    y, sr_loaded = librosa.load(wav_path, sr=sr, mono=True)
    samples_per_segment = int(segment_duration * sr)
    num_segments = len(y) // samples_per_segment

    segments = []
    for i in range(num_segments):
        segment = y[i * samples_per_segment: (i + 1) * samples_per_segment]
        max_abs = np.max(np.abs(segment)) + 1e-9
        normalized = segment / max_abs
        segments.append(normalized.astype(np.float32))

    return np.array(segments, dtype=np.float32)

def extract_features(segment, sr=DEFAULT_FS, n_mfcc=13):
    """
    Extract audio features from a segment:
    - Fundamental frequency (pitch) using librosa.pyin
    - Spectral centroid
    - Spectral rolloff
    - MFCCs (Mel-frequency cepstral coefficients)

    Args:
        segment (np.ndarray): Audio segment samples.
        sr (int): Sample rate.
        n_mfcc (int): Number of MFCC coefficients.

    Returns:
        np.ndarray: Concatenated feature vector.
    """
    # Pitch estimation with librosa.pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(segment,
                                                 fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7'))
    f0_mean = np.nanmean(f0) if np.any(np.isfinite(f0)) else 0.0

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, roll_percent=0.85).mean()

    # MFCC
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc).mean(axis=1)

    feature_vector = np.hstack([f0_mean, spectral_centroid, spectral_rolloff, mfccs]).astype(np.float32)
    return feature_vector

def create_feature_dataset(wav_path, segment_duration=SEGMENT_DURATION, sr=DEFAULT_FS, n_mfcc=13):
    """
    Processes wav file into feature and segment datasets for training.

    Args:
        wav_path (str): WAV file path.
        segment_duration (float): Seconds per segment.
        sr (int): Sampling rate.
        n_mfcc (int): Number of MFCC coefficients.

    Returns:
        tuple: (features, segments)
            features: np.ndarray of shape (num_segments, feature_dim)
            segments: np.ndarray of shape (num_segments, samples_per_segment)
    """
    segments = load_and_segment_audio(wav_path, segment_duration, sr)
    features = np.array([extract_features(seg, sr, n_mfcc) for seg in segments], dtype=np.float32)
    return features, segments
