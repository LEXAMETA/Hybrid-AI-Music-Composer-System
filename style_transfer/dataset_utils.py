# /HybridAI-Music-Composer/style_transfer/dataset_utils.py

import numpy as np
from sklearn.preprocessing import StandardScaler

def merge_datasets(base_features, base_segments, ext_features, ext_segments):
    """
    Concatenates base and external feature and segment arrays vertically.
    """
    merged_features = np.concatenate([base_features, ext_features], axis=0)
    merged_segments = np.concatenate([base_segments, ext_segments], axis=0)
    return merged_features, merged_segments

def normalize_features(features):
    """
    Fits and transforms features using StandardScaler.
    Returns the scaled features and the fitted scaler.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

def validate_dataset_shapes(features, segments):
    """
    Ensures datasets are non-empty and have matching dimensions.
    """
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got {features.ndim}D")
    if segments.ndim != 2:
        raise ValueError(f"Segments must be 2D array, got {segments.ndim}D")
    if len(features) != len(segments):
        raise ValueError(f"Feature and segment counts must be equal: {len(features)} != {len(segments)}")
    if len(features) == 0:
        raise ValueError("Dataset is empty")
