
import numpy as np
from audio_layers.audio_layer import AudioLayer
from style_transfer.dataset_utils import merge_datasets, normalize_features, validate_dataset_shapes
from style_transfer.wav_preprocessing import create_feature_dataset

class AudioLayerExt:
    def __init__(self, base_layer: AudioLayer):
        self.base_layer = base_layer
        self.__dict__.update(base_layer.__dict__)

    def make_synthetic_dataset(self):
        return self.base_layer.make_synthetic_dataset()

    def train_model(self, epochs=8):
        return self.base_layer.train_model(epochs=epochs)

    def train_with_external_data(self, ext_features, ext_segments, epochs=8):
        if self._train_X is None or self._train_Y is None:
            self.make_synthetic_dataset()

        validate_dataset_shapes(self._train_X, self._train_Y)
        validate_dataset_shapes(ext_features, ext_segments)

        if self._train_X.shape[1] != ext_features.shape[1]:
            raise ValueError("Feature dimension mismatch between datasets")
        if self._train_Y.shape[1] != ext_segments.shape[1]:
            raise ValueError("Segment length mismatch between datasets")

        merged_X, merged_Y = merge_datasets(self._train_X, self._train_Y, ext_features, ext_segments)
        scaled_X, scaler = normalize_features(merged_X)

        self._train_X = scaled_X
        self._train_Y = merged_Y
        self.scaler_mean = scaler.mean_
        self.scaler_std = scaler.scale_

        return self.train_model(epochs=epochs)

def style_transfer_train(wav_path: str, base_audio_layer: AudioLayer, epochs=8):
    print(f"Loading WAV and extracting features: {wav_path}")
    ext_features, ext_segments = create_feature_dataset(wav_path)
    print(f"Extracted {len(ext_features)} segments.")

    adapted_layer = AudioLayerExt(base_audio_layer)
    print("Training the adapted AudioLayer model...")
    adapted_layer.train_with_external_data(ext_features, ext_segments, epochs=epochs)
    print("Training complete.")

    return adapted_layer
