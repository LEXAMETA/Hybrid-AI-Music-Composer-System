
import time
import numpy as np
from scipy.io.wavfile import write as write_wav

from audio_layers.audio_layer import AudioLayer
from generators.sequencer import StepSequencer, ArrangementController

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

    output_file = "dubstep_wobble_output.wav"
    write_wav(output_file, fs, (song_audio * 32767).astype(np.int16))
    print(f"Dubstep wobble audio generated and saved to {output_file}")

    print(f"Generation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
