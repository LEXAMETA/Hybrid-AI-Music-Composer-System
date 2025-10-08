# /HybridAI-Music-Composer/style_transfer/style_adapt.py

import numpy as np

# --- MOCK PLACEHOLDERS (Replace with real imports in full system) ---
class AudioLayer:
    def __init__(self, base_freq, pan=0.0, name="layer", num_instincts=3, harmonics=6):
        self.base_freq = base_freq
        self.name = name
        self._train_X = None
        self._train_Y = None
        self.scaler_mean = None
        self.scaler_std = None
        self.model = None

    def make_synthetic_dataset(self, n=100):
        # Mock implementation: base layer features should match external features
        X = np.random.rand(n, 17) # Assuming 1 (f0) + 2 (spec) + 13 (mfcc) = 16 features, plus 3 synth params = 19
        Y = np.random.rand(n, 5512)
        self._train_X = X
        self._train_Y = Y
        return X, Y

    def train_model(self, epochs=8):
        print(f"[{self.name}] Mock training for {epochs} epochs...")
        # In a real system, this would fit the model using self._train_X and self._train_Y
        self.model = object() # Mock model instance
        return self.model
# The other utility functions are mocked or imported from dataset_utils/wav_preprocessing.

from .dataset_utils import merge_datasets, normalize_features, validate_dataset_shapes
from .wav_preprocessing import create_feature_dataset
# ---------------------------------------------------------------------

class AudioLayerExt:
    def __init__(self, base_layer: AudioLayer):
        self.base_layer = base_layer
        # Delegate attributes to the base layer's properties (CRITICAL for Keras/TF access)
        # In a real system, we must ensure base_layer is *already* trained or initialized
        self.__dict__.update(base_layer.__dict__)

    def train_with_external_data(self, ext_features, ext_segments, epochs=8):
        # 1. Ensure Base Layer's synthetic data is loaded/generated
        if self._train_X is None or self._train_Y is None:
            print("[StyleAdapt] Generating synthetic dataset for base layer...")
            self.base_layer.make_synthetic_dataset() # Populates base_layer._train_X/_Y
            self._train_X = self.base_layer._train_X # Copy to self
            self._train_Y = self.base_layer._train_Y # Copy to self
        
        # 2. Validate shapes
        validate_dataset_shapes(self._train_X, self._train_Y)
        validate_dataset_shapes(ext_features, ext_segments)

        # 3. Check feature dimension consistency (CRITICAL)
        if self._train_X.shape[1] != ext_features.shape[1]:
            raise ValueError(f"Feature dimension mismatch: Base={self._train_X.shape[1]}, Ext={ext_features.shape[1]}")
        if self._train_Y.shape[1] != ext_segments.shape[1]:
            raise ValueError(f"Segment length mismatch: Base={self._train_Y.shape[1]}, Ext={ext_segments.shape[1]}")

        # 4. Merge Datasets
        # Note: Merging synthetic features (which include synth params) and external features (which are audio features) 
        # is the main complexity here. For now, we assume the synth features were designed to match the external features' dimension.
        # This will need careful attention in the final AudioLayer implementation.
        merged_X, merged_Y = merge_datasets(self._train_X, self._train_Y, ext_features, ext_segments)

        # 5. Normalize Features
        scaled_X, scaler = normalize_features(merged_X)

        # 6. Update internal state with merged, scaled data and scaler params
        self._train_X = scaled_X
        self._train_Y = merged_Y
        self.scaler_mean = scaler.mean_
        self.scaler_std = scaler.scale_

        # 7. Perform the actual training
        print(f"[StyleAdapt] Merged dataset size: {len(self._train_X)}. Training model...")
        model = self.base_layer.train_model(epochs=epochs) # Calls train_model with merged data/scalers on self

        return model

def style_transfer_train(wav_path: str, base_audio_layer: AudioLayer, epochs=8):
    """
    Orchestrates the style adaptation training process.
    """
    print(f"Loading WAV and extracting features: {wav_path}")
    ext_features, ext_segments = create_feature_dataset(wav_path)
    
    if ext_features.size == 0:
        print("WARNING: No valid segments extracted from WAV. Aborting style transfer.")
        return AudioLayerExt(base_audio_layer) # Return unadapted layer
        
    print(f"Extracted {len(ext_features)} segments.")

    adapted_layer = AudioLayerExt(base_audio_layer)
    print("Training the adapted AudioLayer model...")
    adapted_layer.train_with_external_data(ext_features, ext_segments, epochs=epochs)
    print("Training complete.")

    return adapted_layer
