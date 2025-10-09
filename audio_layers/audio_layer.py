# /HybridAI-Music-Composer/audio_layers/audio_layer.py (CRITICAL FIX: Filter Stability)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer 
from tensorflow.keras.callbacks import Callback
from scipy.signal import butter, lfilter
import random
# 🐛 CRITICAL FIX: Import the complex model builder
from .decoder_models import build_decoder_with_instincts
from .train_helpers import CyclicLR 

# Audio constants
fs = 22050
step_duration = 0.25
samples_per_step = int(fs * step_duration)
step_dur = step_duration 

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

    def make_synthetic_dataset(self, n=1000): 
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
            
            # Ensure features are float32
            features = np.concatenate([[timbre_shift, fm_freq, fm_index], harmonic_weights])
            X.append(features.astype(np.float32))
            Y.append(wave.astype(np.float32))

        self._train_X = np.array(X, dtype=np.float32)
        self._train_Y = np.array(Y, dtype=np.float32)
        return self._train_X, self._train_Y

    def train_model(self, epochs=50, batch_size=32, verbose=1, lr=1e-4): 
        if self._train_X is None or self._train_Y is None:
            self.make_synthetic_dataset(n=1000)

        # Ensure features are scaled correctly
        self.scaler_mean = np.mean(self._train_X, axis=0)
        self.scaler_std = np.std(self._train_X, axis=0) + 1e-9
        train_X = (self._train_X - self.scaler_mean) / self.scaler_std

        # 🐛 CRITICAL FIX: Use the imported, complex decoder model
        self.model = build_decoder_with_instincts(
            input_dim_raw=train_X.shape[1], 
            num_instincts=self.num_instincts, 
            output_len=self._train_Y.shape[1], 
            lr=lr
        )

        clr = CyclicLR(base_lr=1e-5, max_lr=lr * 10, step_size=200)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) 

        history = self.model.fit(train_X, self._train_Y, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.1, callbacks=[clr, early_stop], verbose=verbose)

        val_loss = history.history['val_loss'][-1]
        print(f"[{self.name}] Training completed. Final val_loss={val_loss:.5f}")

        return self.model

    def precompute_predictions_for_song(self, total_steps):
        if self.model is None:
            print(f"[{self.name}] WARNING: Model is not trained. Cannot precompute.")
            return None
        
        features = []
        # ... (feature generation logic unchanged)
        for i in range(total_steps):
            seed = int((self.base_freq * 1000 + i) % 2**31)
            rnd = np.random.RandomState(seed)
            timbre_shift = 0.0 # Fixed timbre for playback
            fm_freq = 5.0      # Fixed FM params for playback
            fm_index = 6.0
            harmonic_weights = rnd.dirichlet(np.ones(self.harmonics))
            feat = np.concatenate([[timbre_shift, fm_freq, fm_index], harmonic_weights])
            features.append(feat)
            
        features = np.array(features, dtype=np.float32)
        features_norm = (features - self.scaler_mean) / self.scaler_std if self.scaler_mean is not None else features
        
        preds = self.model.predict(features_norm, batch_size=128)
        preds = np.nan_to_num(preds)
        
        raw_max_abs = np.max(np.abs(preds))
        print(f"[{self.name}] Precomp Raw Max Amp: {raw_max_abs:.6f}") 
        
        max_abs = raw_max_abs + 1e-9
        
        if max_abs < 0.02:
            preds = preds / max_abs * 0.2
            print(f"[{self.name}] Rescued low-amp signal. New max amp: {np.max(np.abs(preds)):.6f}")

        self._precomputed = preds
        return self._precomputed

    def get_precomputed_step(self, step_idx):
        if self._precomputed is None:
            # Return silence instead of None to prevent mixer errors
            return np.zeros((samples_per_step, 2), dtype=np.float32)
            
        w = self._precomputed[step_idx % self._precomputed.shape[0]]
        
        # Simple panning automation
        pan_mod = np.sin(step_idx * 0.03) * self.pan 
        left = w * (1 - pan_mod) / 2.0
        right = w * (1 + pan_mod) / 2.0
        stereo = np.vstack([left, right]).T.astype(np.float32)
        return stereo

    @staticmethod
    def generate_textured_sound(base_freq, harmonic_weights, dur, fm_freq, fm_index, timbre_shift):
        t = np.linspace(0, dur, int(fs * dur), endpoint=False)
        fm_mod = fm_index * np.sin(2 * np.pi * fm_freq * t)
        signal_wave = np.zeros_like(t, dtype=np.float32) # Explicitly float32
        for h, w in enumerate(harmonic_weights, start=1):
            # Ensure harmonic calculation is stable
            signal_wave += w * np.sin(2 * np.pi * (base_freq * h + fm_mod) * t).astype(np.float32)
            
        # Envelope generation
        envelope = np.ones_like(t, dtype=np.float32)
        attack = int(len(t) * 0.005)
        release = int(len(t) * 0.02)
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack).astype(np.float32)
        if release > 0:
            envelope[-release:] = np.linspace(1, 0, release).astype(np.float32)
        signal_wave *= envelope
        
        # Noise generation
        noise = np.random.normal(0, 0.02, signal_wave.shape).astype(np.float32)
        filtered_noise = AudioLayer.bandpass_filter(noise, max(20, base_freq * 0.4), base_freq * 3.0, fs)
        
        # Combined signal
        combined = signal_wave + 0.2 * filtered_noise 
        
        waveform = combined.astype(np.float32)
        
        # Final Normalization
        max_abs = np.max(np.abs(waveform)) + 1e-9
        waveform /= max_abs 
        
        return waveform

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        
        # 🐛 CRITICAL FIX: Ensure normalized frequencies are strictly within (0, 1)
        low = np.clip(lowcut / nyq, 1e-6, 0.999) 
        high = np.clip(highcut / nyq, low + 1e-5, 0.999) # Ensure high > low
        
        if low >= high: # Safety fallback
             return data * 0.0
             
        b, a = butter(order, [low, high], btype='band')
        
        # lfilter without zi is safe here as this is a new block of noise
        return lfilter(b, a, data)
