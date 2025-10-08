# /HybridAI-Music-Composer/scripts/run_pipeline.py

import time
import os
import numpy as np
from scipy.io.wavfile import write as write_wav

# Assuming the corrected imports:
from audio_layers.audio_layer import AudioLayer 
from generators.ga_composer import GeneticMusicGenerator, chromosome_to_pitch_sequence
from generators.sequencer import StepSequencer, ArrangementController
# Mock style_transfer imports as they weren't provided in full:
class AudioLayerExt(AudioLayer): 
    def train_with_external_data(self, ext_feats, ext_segments):
        # Placeholder for actual training call using external data
        print(f"[{self.name}] Training adapted model with {len(ext_feats)} external segments.")
        self.model = MockKerasModel() # Assume training sets a model
        # The training logic from style_adapt.py would be called here
        
class MockKerasModel:
    def __call__(self, input_tensor):
        # Mock evaluation: returns a random score
        return tf.constant(np.random.rand(1), dtype=tf.float32)

def build_external_dataset(path):
    # Mock: returns empty features/segments
    return np.array([[]]), np.array([[]])

# Mock display utility
def display_audio_and_links(files):
    print(f"Displaying/playing generated files: {files}")

# Constants matching rest of codebase:
fs = 22050
STEP_DUR = 0.25
SAMPLES_PER_STEP = int(fs * STEP_DUR)
SEQUENCE_LENGTH = 64 # Use the same length as dubstep generation for consistency

# Use TensorFlow only if needed for GeneticMusicGenerator evaluator
try:
    import tensorflow as tf
except ImportError:
    print("WARNING: TensorFlow not found. GeneticMusicGenerator may fail.")
    tf = None

def main_pipeline():
    print("=== 1. Generate baseline dubstep_wobble_output.wav ===")
    start = time.time()
    # 🐛 FIX: Call the generate_dubstep.py main function directly if possible, 
    # otherwise use os.system (assuming the file is executable and in the path)
    if os.path.exists("scripts/generate_dubstep.py"):
         os.system("python scripts/generate_dubstep.py")
    else:
         print("WARNING: scripts/generate_dubstep.py not found. Skipping baseline generation.")
         
    dubstep_wav = "dubstep_wobble_output.wav"
    if not os.path.exists(dubstep_wav):
        # Create a silent file if generation fails to prevent crashes
        write_wav(dubstep_wav, fs, np.zeros(fs, dtype=np.int16))
    print(f"Done in {time.time() - start:.2f} seconds")

    print("=== 2. Train adapted AudioLayer on dubstep output ===")
    base_layer = AudioLayer(base_freq=60.0, pan=0.0, name="bass")
    ext_feats, ext_segments = build_external_dataset(dubstep_wav) # Load data
    
    # You must initialize the base layer's model *before* passing it to AudioLayerExt
    # In a real system, you'd call: base_layer.train_model() 

    adapted_layer = AudioLayerExt(base_layer)
    adapted_layer.train_with_external_data(ext_feats, ext_segments)
    print("Adapted AudioLayer trained.")

    print("=== 3. Run GA composition with adapted decoder ===")
    # Requires a Keras/TensorFlow model for evaluation
    if not tf:
        print("Skipping GA: TensorFlow is not imported/available.")
        return
        
    ga = GeneticMusicGenerator(evaluator_model=adapted_layer.model, 
                               population_size=30, 
                               sequence_length=SEQUENCE_LENGTH)
                               
    best_chromo = ga.evolve(generations=20)
    
    print("Best chromosome evolved.")

    print("=== 4. Precompute predictions for best sequence ===")
    adapted_layer.precompute_predictions_for_song(len(best_chromo))

    print("=== 5. Setup sequencer and arrangement ===")
    sequencer = StepSequencer([adapted_layer], pattern_length=len(best_chromo))
    arrangement = ArrangementController(sequencer, total_steps=len(best_chromo))

    section_plan = [("Full song", len(best_chromo))]
    final_audio = arrangement.generate_song(section_plan)

    output_file = "final_adapted_mix.wav"
    write_wav(output_file, fs, (final_audio * 32767).astype(np.int16))
    print(f"Final mix saved to {output_file}")

    print("=== 6. Display and playback generated audio files ===")
    all_files = [dubstep_wav, output_file]
    display_audio_and_links(all_files)
    print("Pipeline complete.")

if __name__ == "__main__":
    main_pipeline()
