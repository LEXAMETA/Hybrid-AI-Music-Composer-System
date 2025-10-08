
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class CyclicLR(tf.keras.callbacks.Callback):
    """
    Cyclical learning rate scheduler.
    """
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
    """
    Create a synthetic dataset for training based on current AudioLayer parameters.

    Returns:
        tuple: features (X), waveforms (Y)
    """
    X = []
    Y = []
    for _ in range(n_samples):
        # Random timbre shift and FM parameters within reasonable ranges
        timbre_shift = np.random.uniform(-0.45, 0.45)
        fm_freq = np.random.uniform(3.0, 8.0)
        fm_index = np.random.uniform(2.0, 12.0)

        # Harmonic weights as Dirichlet distribution
        harmonic_weights = np.random.dirichlet(np.ones(audio_layer.harmonics))

        wave = audio_layer.generate_textured_sound(
            audio_layer.base_freq * (1.0 + timbre_shift),
            harmonic_weights,
            0.25,
            fm_freq,
            fm_index,
            timbre_shift
        )
        wave /= (np.max(np.abs(wave)) + 1e-9)

        features = np.concatenate([[timbre_shift, fm_freq, fm_index], harmonic_weights])
        X.append(features.astype(np.float32))
        Y.append(wave.astype(np.float32))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def scale_features(features, scaler=None):
    """
    Scale features using provided StandardScaler or new scaler.

    Args:
        features (np.ndarray): Feature matrix.
        scaler (StandardScaler): Optional pre-fitted scaler.

    Returns:
        tuple: (scaled_features, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = scaler.transform(features)
    return scaled_features, scaler

def train_audio_layer_model(audio_layer, epochs=30, batch_size=32, verbose=1):
    """
    Train the AudioLayer's neural decoder on its synthetic or combined datasets.

    Args:
        audio_layer (AudioLayer): The AudioLayer instance.
        epochs (int): Training epochs.
        batch_size (int): Mini-batch size.
        verbose (int): Verbosity.

    Returns:
        tf.keras.Model: The trained model.
    """
    if audio_layer._train_X is None or audio_layer._train_Y is None:
        audio_layer._train_X, audio_layer._train_Y = create_synthetic_dataset(audio_layer)

    # Scale features
    scaled_X, scaler = scale_features(audio_layer._train_X)
    audio_layer.scaler_mean = scaler.mean_
    audio_layer.scaler_std = scaler.scale_

    audio_layer.model = audio_layer.model or audio_layer.build_model(
        input_dim=scaled_X.shape[1], output_len=audio_layer._train_Y.shape[1])

    # Setup the cyclic learning rate scheduler
    clr_cb = CyclicLR(base_lr=1e-5, max_lr=1e-3, step_size=200)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

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
