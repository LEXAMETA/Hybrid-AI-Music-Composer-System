import numpy as np
# Assuming the existence of these external modules
# from audio_layers.audio_layer import AudioLayer
# from style_transfer.dataset_utils import merge_datasets, normalize_features, validate_dataset_shapes
# from style_transfer.wav_preprocessing import create_feature_dataset

# Placeholder definitions for classes/functions not provided, necessary for execution
class AudioLayer:
    def __init__(self, base_freq, pan=0.0, name="layer", num_instincts=3, harmonics=6):
        self.base_freq = base_freq
        self.name = name
        self._train_X = np.zeros((1, 10)) # Placeholder
        self._train_Y = np.zeros((1, 5512)) # Placeholder
        self.scaler_mean = None
        self.scaler_std = None
        self.model = None

    def make_synthetic_dataset(self):
        # Mock implementation
        self._train_X = np.random.rand(100, 10)
        self._train_Y = np.random.rand(100, 5512)
        return self._train_X, self._train_Y

    def train_model(self, epochs=8):
        print(f"[AudioLayer] Mock training for {epochs} epochs...")
        return None # Mock model

    def generate_adapted(self, features): # Mock method needed for the debug loop
        # Mock audio generation - should return a segment of audio
        return np.random.rand(features.shape[0], self._train_Y.shape[1]) * 2 - 1

def merge_datasets(X1, Y1, X2, Y2): return np.concatenate([X1, X2]), np.concatenate([Y1, Y2])
def normalize_features(X): 
    class Scaler:
        def __init__(self, mean, scale): self.mean_ = mean; self.scale_ = scale
    return X, Scaler(X.mean(axis=0), X.std(axis=0) + 1e-9)
def validate_dataset_shapes(X, Y): pass
def create_feature_dataset(wav_path): return np.random.rand(50, 10), np.random.rand(50, 5512)


class AudioLayerExt:
    def __init__(self, base_layer: AudioLayer):
        self.base_layer = base_layer
        # Directly copy attributes from the base_layer to AudioLayerExt instance
        # Note: This is generally discouraged for encapsulation, but follows the original snippet's pattern
        self.__dict__.update(base_layer.__dict__)

    def make_synthetic_dataset(self):
        # Calls the base layer's dataset generation but updates self's attributes
        X, Y = self.base_layer.make_synthetic_dataset()
        self._train_X = X
        self._train_Y = Y
        return X, Y

    def train_model(self, epochs=8):
        # Assuming the base layer's train_model uses the data/scalers set on self (via __dict__.update)
        return self.base_layer.train_model(epochs=epochs)

    def train_with_external_data(self, ext_features, ext_segments, epochs=8):
        if self._train_X is None or self._train_Y is None:
            # Need to call make_synthetic_dataset on the *extended* layer to correctly set _train_X/_Y
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

        # 1. Perform the actual training
        model = self.train_model(epochs=epochs)
        
        # 2. --- Merged Debugging/Logging Code (Post-Training Check) ---
        print("\n--- Style Adaptation Post-Training Check ---")
        
        # Use a small set of the merged features for a quick generation check
        test_features = merged_X[:5]
        
        # Assuming generate_adapted takes features and returns a waveform
        try:
            # The base_layer's methods are available via self due to __dict__.update
            adapted_waveform = self.generate_adapted(test_features) 
            
            # Print the stats for each requested epoch check
            # Since Keras training is done, we'll just check the result once.
            for i in range(min(5, adapted_waveform.shape[0])):
                 print(f"[StyleAdapt] Sample {i} adapted segment - min: {adapted_waveform[i].min():.4f}, max: {adapted_waveform[i].max():.4f}, mean: {adapted_waveform[i].mean():.4f}")
            
        except AttributeError:
            print("[StyleAdapt] WARNING: 'generate_adapted' method not found on base layer. Skipping waveform check.")

        print("--- End Post-Training Check ---\n")
        # 3. Return the trained model (or None if mock)
        return model

def style_transfer_train(wav_path: str, base_audio_layer: AudioLayer, epochs=8):
    print(f"Loading WAV and extracting features: {wav_path}")
    ext_features, ext_segments = create_feature_dataset(wav_path)
    print(f"Extracted {len(ext_features)} segments.")

    adapted_layer = AudioLayerExt(base_audio_layer)
    print("Training the adapted AudioLayer model...")
    adapted_layer.train_with_external_data(ext_features, ext_segments, epochs=epochs)
    print("Training complete.")

    return adapted_layer
