
import numpy as np
from sklearn.preprocessing import StandardScaler

def merge_datasets(base_features, base_segments, ext_features, ext_segments):
    merged_features = np.concatenate([base_features, ext_features], axis=0)
    merged_segments = np.concatenate([base_segments, ext_segments], axis=0)
    return merged_features, merged_segments

def normalize_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

def validate_dataset_shapes(features, segments):
    if features.ndim != 2:
        raise ValueError("Features must be 2D array")
    if segments.ndim != 2:
        raise ValueError("Segments must be 2D array")
    if len(features) != len(segments):
        raise ValueError("Feature and segment counts must be equal")
    if len(features) == 0:
        raise ValueError("Dataset is empty")
