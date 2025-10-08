# /HybridAI-Music-Composer/scripts/generate_dubstep.py

import time
import numpy as np
from scipy.io.wavfile import write as write_wav

# Placeholder Imports (In a real system, these would be:
# from audio_layers.audio_layer import AudioLayer
# from generators.sequencer import StepSequencer, ArrangementController
# We use the provided mock classes/definitions for self-contained execution)

# ===============================================
# PLACEHOLDER CLASSES (Replace with actual imports)
# ===============================================
class AudioLayer:
    """Mock/Simplified AudioLayer with no training or synthesis logic."""
    def __init__(self, base_freq, pan=0.0, name="layer", num_instincts=3, harmonics=6):
        self.base_freq = base_freq
        self.name = name
        self.num_instincts = num_instincts
        self.harmonics = harmonics
        self._precomputed = None

    def precompute_predictions_for_song(self, total_steps):
        """Mocks precomputation by creating random, normalized stereo audio."""
        samples = SAMPLES_PER_STEP
        mock_audio = np.random.uniform(-0.5, 0.5, size=(total_steps, samples, 2)).astype(np.float32)
        self._precomputed = mock_audio
        return self._precomputed

    def get_precomputed_step(self, step_idx):
        if self._precomputed is None:
            return np.zeros((SAMPLES_PER_STEP, 2), dtype=np.float32)
        step_audio = self._precomputed[step_idx % self._precomputed.shape[0]]
        return step_audio

fs = 22050
STEP_DUR = 0.25
SAMPLES_PER_STEP = int(fs * STEP_DUR)

class StepSequencer:
    def __init__(self, layers, pattern_length):
        self.layers = layers
        self.pattern_length = pattern_length
        self.current_step = 0

    def synth_next_step(self):
        stereo_buffer = np.zeros((SAMPLES_PER_STEP, 2), dtype=np.float32)
        for layer in self.layers:
            step_audio = layer.get_precomputed_step(self.current_step)
            # 🐛 FIX: Removed redundant normalization here. Mixing should happen before scaling.
            stereo_buffer += step_audio

        # Clamp to [-1, 1] to prevent hard clipping during accumulation
        stereo_buffer = np.clip(stereo_buffer, -1.0, 1.0)
        
        self.current_step = (self.current_step + 1) % self.pattern_length
        return stereo_buffer

class ArrangementController:
    def __init__(self, sequencer, total_steps):
        self.sequencer = sequencer
        self.total_steps = total_steps
        self.current_step = 0

    def generate_song(self, section_plan):
        full_audio = []
        for section_name, num_steps in section_plan:
            for step_in_section in range(num_steps):
                step_audio = self.sequencer.synth_next_step()
                self.current_step += 1
                if 'build-up' in section_name.lower():
                    volume_scale = 0.5 + (step_in_section / num_steps) * 0.5
                    step_audio *= volume_scale
                full_audio.append(step_audio)
        
        song = np.concatenate(full_audio, axis=0)
        
        # Final track normalization: find the max value and scale everything down
        max_val = np.max(np.abs(song)) + 1e-9
        normalized_song = (song / max_val).astype(np.float32)
        
        print(f"[ArrangementController] Final song shape: {normalized_song.shape}")
        return normalized_song
# ===============================================
# END PLACEHOLDER CLASSES
# ===============================================

def main():
    start_time = time.time()
    print(f"--- Starting Dubstep Generation at 140 BPM ---")

    # In a real run, you'd likely call layer.train_model() here
    layers = [
        AudioLayer(base_freq=60.0, pan=0.0, name="kick", num_instincts=3, harmonics=2),
        AudioLayer(base_freq=200.0, pan=0.0, name="snare", num_instincts=3, harmonics=3),
        AudioLayer(base_freq=30.0, pan=-0.6, name="sub_bass", num_instincts=3, harmonics=7),
        AudioLayer(base_freq=440.0, pan=0.6, name="lead", num_instincts=3, harmonics=6),
        AudioLayer(base_freq=330.0, pan=0.0, name="pad", num_instincts=3, harmonics=6),
    ]

    pattern_length = 64  # 4 bars at 16 steps/bar

    for layer in layers:
        print(f"Initialized layer: {layer.name}")
        # Assuming training happened before this point in a real system
        layer.precompute_predictions_for_song(pattern_length) 

    sequencer = StepSequencer(layers, pattern_length=pattern_length)
    arrangement = ArrangementController(sequencer, total_steps=pattern_length)

    section_plan = [("Full Song", pattern_length)]
    print("Generating song...")
    song_audio = arrangement.generate_song(section_plan)

    # --- Merged Debugging/Logging Code ---
    print(f"[generate_dubstep] Final song_audio shape: {song_audio.shape}")
    print(f"[generate_dubstep] Audio data statistics - min: {song_audio.min():.4f}, max: {song_audio.max():.4f}, mean: {song_audio.mean():.4f}")
    print(f"[generate_dubstep] Audio contains non-zero samples: {np.any(song_audio != 0)}")
    # --- End Merged Code ---

    output_file = "dubstep_wobble_output.wav"
    # Scale to 16-bit PCM integer range
    write_wav(output_file, fs, (song_audio * 32767).astype(np.int16))
    print(f"Dubstep wobble audio generated and saved to {output_file}")

    print(f"Generation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
