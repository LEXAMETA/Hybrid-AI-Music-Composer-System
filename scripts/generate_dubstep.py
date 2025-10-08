import time
import numpy as np
from scipy.io.wavfile import write as write_wav

# Placeholder imports (assuming these classes were defined in previous snippets)
class AudioLayer:
    def __init__(self, base_freq, pan=0.0, name="layer", num_instincts=3, harmonics=6):
        self.base_freq = base_freq
        self.name = name
        self.num_instincts = num_instincts
        self.harmonics = harmonics
        self._precomputed = None

    def precompute_predictions_for_song(self, total_steps):
        # Mock implementation: create a random array for precomputed audio
        steps = total_steps
        samples = 5512  # Assuming SAMPLES_PER_STEP from previous context
        # Generate some mock stereo audio
        mock_audio = np.random.uniform(-0.5, 0.5, size=(steps, samples, 2)).astype(np.float32)
        self._precomputed = mock_audio
        return self._precomputed

    def get_precomputed_step(self, step_idx):
        if self._precomputed is None:
            return np.zeros((5512, 2), dtype=np.float32)
        return self._precomputed[step_idx % self._precomputed.shape[0]]

# Audio constants needed for StepSequencer/ArrangementController
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
            if step_audio is not None:
                stereo_buffer += step_audio
        self.current_step = (self.current_step + 1) % self.pattern_length
        stereo_buffer = np.clip(stereo_buffer, -1.0, 1.0)
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
        max_val = np.max(np.abs(song)) + 1e-9
        return (song / max_val).astype(np.float32)
# End of placeholder imports

fs = 22050
STEP_DUR = 0.25
SAMPLES_PER_STEP = int(fs * STEP_DUR)

def main():
    start_time = time.time()
    print(f"--- Starting Dubstep Generation at 140 BPM ---")

    # Initialize layers
    layers = [
        AudioLayer(base_freq=60.0, pan=0.0, name="kick", num_instincts=3, harmonics=2),
        AudioLayer(base_freq=200.0, pan=0.0, name="snare", num_instincts=3, harmonics=3),
        AudioLayer(base_freq=30.0, pan=-0.6, name="sub_bass", num_instincts=3, harmonics=7),
        AudioLayer(base_freq=440.0, pan=0.6, name="lead", num_instincts=3, harmonics=6),
        AudioLayer(base_freq=330.0, pan=0.0, name="pad", num_instincts=3, harmonics=6),
    ]
    for layer in layers:
        print(f"Initialized layer: {layer.name}")

    pattern_length = 64  # 4 bars at 16 steps/bar

    # Optionally precompute synthesis for layers here:
    for layer in layers:
        layer.precompute_predictions_for_song(pattern_length)

    sequencer = StepSequencer(layers, pattern_length=pattern_length)
    arrangement = ArrangementController(sequencer, total_steps=pattern_length)

    # Simple full-song section plan
    section_plan = [("Full Song", pattern_length)]

    print("Generating song...")
    song_audio = arrangement.generate_song(section_plan)

    # --- Merged Debugging/Logging Code ---
    print(f"[generate_dubstep] Final song_audio shape: {song_audio.shape}")
    print(f"[generate_dubstep] Audio data statistics - min: {song_audio.min():.4f}, max: {song_audio.max():.4f}, mean: {song_audio.mean():.4f}")
    print(f"[generate_dubstep] Audio contains non-zero samples: {np.any(song_audio != 0)}")
    # --- End Merged Code ---

    output_file = "dubstep_wobble_output.wav"
    # Ensure the audio is scaled to 16-bit integer range before saving
    write_wav(output_file, fs, (song_audio * 32767).astype(np.int16))
    print(f"Dubstep wobble audio generated and saved to {output_file}")

    print(f"Generation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
