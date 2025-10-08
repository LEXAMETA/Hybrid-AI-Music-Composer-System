# /Hybrid-AI-Music-Composer/audio_layers/train_helpers.py

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# Assuming build_decoder_with_instincts is available via AudioLayer or an explicit import if this file runs independently

class CyclicLR(tf.keras.callbacks.Callback):
    # ... (CyclicLR implementation remains the same)
    def __init__(self, base_lr=1e-5, max_lr=1e-3, step_size=1000):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iterations = 0

    def clr(self):
        cycle = np.floor(1 + self.iterations / (2 * self.step_size))
        x = abs(self.iterations / self.step_size - 2 * cycle + 1)
        scale = max(0, 1 - x)
        return self.base_lr + (self.max_lr - self.base_lr) * scale

    def on_train_batch_begin(self, batch, logs=None):
        self.iterations += 1
        lr = self.clr()
        try:
            self.model.optimizer.learning_rate.assign(lr)
        except Exception:
            from tensorflow.keras import backend as K
            K.set_value(self.model.optimizer.lr, lr)

def create_synthetic_dataset(audio_layer, n_samples=400):
    # ... (Function body remains the same, relies on AudioLayer.generate_textured_sound)
    X = []
    Y = []
    for _ in range(n_samples):
        timbre_shift = np.random.uniform(-0.45, 0.45)
        fm_freq = np.random.uniform(3.0, 8.0)
        fm_index = np.random.uniform(2.0, 12.0)
        harmonic_weights = np.random.dirichlet(np.ones(audio_layer.harmonics))
        wave = audio_layer.generate_textured_sound(
            audio_layer.base_freq * (1.0 + timbre_shift),
            harmonic_weights,
            0.25,
            fm_freq,
            fm_index,
            timbre_shift
        )
        # Normalization is handled inside generate_textured_sound
        features = np.concatenate([[timbre_shift, fm_freq, fm_index], harmonic_weights])
        X.append(features.astype(np.float32))
        Y.append(wave.astype(np.float32))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def scale_features(features, scaler=None):
    # ... (Function body remains the same)
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = scaler.transform(features)
    return scaled_features, scaler

def train_audio_layer_model(audio_layer, epochs=30, batch_size=32, verbose=1):
    """
    Train the AudioLayer's neural decoder on its synthetic or combined datasets.
    """
    if audio_layer._train_X is None or audio_layer._train_Y is None:
        audio_layer._train_X, audio_layer._train_Y = create_synthetic_dataset(audio_layer)

    # Scale features
    scaled_X, scaler = scale_features(audio_layer._train_X)
    audio_layer.scaler_mean = scaler.mean_
    audio_layer.scaler_std = scaler.scale_

    # 🐛 Model building relies on a helper method that needs to be implemented or rely on audio_layer.train_model
    # For compatibility, we rely on the internal logic now correctly pointing to the right model in audio_layer.py
    if audio_layer.model is None:
         # Calling the internal method which now points to the complex decoder
         audio_layer.model = audio_layer.train_model(epochs=1, verbose=0, lr=1e-4) 

    # Setup the cyclic learning rate scheduler
    clr_cb = CyclicLR(base_lr=1e-5, max_lr=1e-3, step_size=200)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Final fit (using scaled_X, _train_Y)
    history = audio_layer.model.fit(
        scaled_X,
        audio_layer._train_Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[clr_cb, early_stop],
        verbose=verbose
    )
    return audio_layer.model
