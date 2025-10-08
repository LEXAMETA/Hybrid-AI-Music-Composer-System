import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Reshape, Conv1D, LSTM, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from scipy.signal import butter, lfilter
import random

# Audio constants
fs = 22050  # Sample rate consistent with system
step_dur = 0.25
samples_per_step = int(fs * step_dur)

# CyclicLR callback for training learning rate schedules
class CyclicLR(Callback):
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

# RecurrentAttention Layer implementation (can replace with your own)
class RecurrentAttention(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        self.V = self.add_weight(shape=(self.units, 1), initializer='glorot_uniform', trainable=True)
        super().build(input_shape)
    def call(self, x):
        h = tf.tanh(tf.tensordot(x, self.W, axes=1))
        score = tf.tensordot(h, self.V, axes=1)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * x, axis=1)
        return context

def combined_time_freq_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    # Add frequency domain loss if you want here
    return mse

# Decoder builder function
def build_decoder_with_instincts(input_dim, num_instincts=3, output_len=samples_per_step, lr=1e-4):
    inp = Input(shape=(input_dim,), name='features')
    x = Dense(128, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)
    x = Dense(128 * 8, activation='relu')(x)
    x = Reshape((128, 8))(x)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    lstm_out = LSTM(128, return_sequences=True, dropout=0.05)(x)
    attention = RecurrentAttention(64)(lstm_out)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-5))(attention)
    x = BatchNormalization()(x)
    out = Dense(output_len, activation='tanh')(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0), loss=combined_time_freq_loss)
    return model

# AudioLayer class
class AudioLayer:
    def __init__(self, base_freq, pan=0.0, name="layer", num_instincts=3, harmonics=6):
        self.base_freq = float(base_freq)
        self.pan = float(pan)
        self.name = name
        self.num_instincts = num_instincts
        self.harmonics = harmonics

        self._train_X = None
        self._train_Y = None
        self.scaler_mean = None
        self.scaler_std = None
        self.model = None
        self._precomputed = None

    def make_synthetic_dataset(self, n=400):
        X = []
        Y = []
        for _ in range(n):
            timbre_shift = np.random.uniform(-0.45, 0.45)
            fm_freq = np.random.uniform(3.0, 8.0)
            fm_index = np.random.uniform(2.0, 12.0)
            harmonic_weights = np.random.dirichlet(np.ones(self.harmonics))
            wave = self.generate_textured_sound(
                self.base_freq * (1.0 + timbre_shift),
                harmonic_weights, step_dur, fm_freq, fm_index, timbre_shift)
            wave /= (np.max(np.abs(wave)) + 1e-9)
            features = np.concatenate([[timbre_shift, fm_freq, fm_index], harmonic_weights])
            X.append(features.astype(np.float32))
            Y.append(wave.astype(np.float32))

        self._train_X = np.array(X, dtype=np.float32)
        self._train_Y = np.array(Y, dtype=np.float32)
        return self._train_X, self._train_Y

    def train_model(self, epochs=30, batch_size=32, verbose=1, lr=1e-4):
        if self._train_X is None or self._train_Y is None:
            self.make_synthetic_dataset(n=400)

        self.scaler_mean = np.mean(self._train_X, axis=0)
        self.scaler_std = np.std(self._train_X, axis=0) + 1e-9
        train_X = (self._train_X - self.scaler_mean) / self.scaler_std

        self.model = build_decoder_with_instincts(
            input_dim=train_X.shape[1], num_instincts=self.num_instincts, output_len=self._train_Y.shape[1], lr=lr)

        clr = CyclicLR(base_lr=1e-5, max_lr=lr * 10, step_size=200)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(train_X, self._train_Y, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.1, callbacks=[clr, early_stop], verbose=verbose)

        val_loss = history.history['val_loss'][-1]
        print(f"[{self.name}] Training completed. Final val_loss={val_loss:.5f}")

        return self.model

    def precompute_predictions_for_song(self, total_steps):
        if self.model is None:
            return None
        features = []
        for i in range(total_steps):
            seed = int((self.base_freq * 1000 + i) % 2**31)
            rnd = np.random.RandomState(seed)
            timbre_shift = 0.0
            fm_freq = 5.0
            fm_index = 6.0
            harmonic_weights = rnd.dirichlet(np.ones(self.harmonics))
            feat = np.concatenate([[timbre_shift, fm_freq, fm_index], harmonic_weights])
            features.append(feat)
        features = np.array(features, dtype=np.float32)
        features_norm = (features - self.scaler_mean) / self.scaler_std if self.scaler_mean is not None else features
        preds = self.model.predict(features_norm, batch_size=128)
        preds = np.nan_to_num(preds)
        max_abs = np.max(np.abs(preds)) + 1e-9
        if max_abs < 0.02:
            preds = preds / max_abs * 0.2
        self._precomputed = preds
        return self._precomputed

    def get_precomputed_step(self, step_idx):
        if self._precomputed is None:
            return None
        w = self._precomputed[step_idx % self._precomputed.shape[0]]
        pan_mod = np.sin(step_idx * 0.03) * 0.4
        left = w * (1 - pan_mod) / 2.0
        right = w * (1 + pan_mod) / 2.0
        stereo = np.vstack([left, right]).T.astype(np.float32)
        return stereo

    @staticmethod
    def generate_textured_sound(base_freq, harmonic_weights, dur, fm_freq, fm_index, timbre_shift):
        t = np.linspace(0, dur, int(fs * dur), endpoint=False)
        fm_mod = fm_index * np.sin(2 * np.pi * fm_freq * t)
        signal_wave = np.zeros_like(t)
        for h, w in enumerate(harmonic_weights, start=1):
            signal_wave += w * np.sin(2 * np.pi * (base_freq * h + fm_mod) * t)
        envelope = np.ones_like(t)
        attack = int(len(t) * 0.005)
        release = int(len(t) * 0.02)
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        if release > 0:
            envelope[-release:] = np.linspace(1, 0, release)
        signal_wave *= envelope
        noise = np.random.normal(0, 0.02, signal_wave.shape)
        filtered_noise = AudioLayer.bandpass_filter(noise, max(20, base_freq * 0.4), base_freq * 3.0, fs)
        combined = signal_wave + 0.5 * filtered_noise
        combined /= (np.max(np.abs(combined)) + 1e-9)
        return combined.astype(np.float32)

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 1e-6)
        high = min(highcut / nyq, 0.999)
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)
