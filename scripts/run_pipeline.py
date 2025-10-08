
import time
import os
from scipy.io.wavfile import write as write_wav

from audio_layers.audio_layer import AudioLayer
from generators.ga_composer import GeneticMusicGenerator, chromosome_to_pitch_sequence
from generators.sequencer import StepSequencer, ArrangementController
from style_transfer.style_adapt import AudioLayerExt, build_external_dataset
from utils.display_utils import display_audio_and_links  # Your player utility

# Constants matching rest of codebase:
fs = 22050
STEP_DUR = 0.25
SAMPLES_PER_STEP = int(fs * STEP_DUR)
TOTAL_STEPS = 1024  # e.g. ~4 minutes at 0.25 step duration

def main_pipeline():
    print("=== 1. Generate baseline dubstep_wobble_output.wav ===")
    start = time.time()
    os.system("python scripts/generate_dubstep.py")  # or call a main func directly
    dubstep_wav = "dubstep_wobble_output.wav"
    if not os.path.exists(dubstep_wav):
        raise RuntimeError(f"Dubstep wav failed to generate: {dubstep_wav}")
    print(f"Done in {time.time() - start:.2f} seconds")

    print("=== 2. Train adapted AudioLayer on dubstep output ===")
    base_layer = AudioLayer(base_freq=60.0, pan=0.0, name="bass")
    ext_feats, ext_segments = build_external_dataset(dubstep_wav)
    adapted_layer = AudioLayerExt(base_layer)
    adapted_layer.train_with_external_data(ext_feats, ext_segments)
    print("Adapted AudioLayer trained.")

    print("=== 3. Run GA composition with adapted decoder ===")
    ga = GeneticMusicGenerator(evaluator_model=adapted_layer.model, population_size=30)
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
    write_wav(output_file, fs, (final_audio * 32767).astype('int16'))
    print(f"Final mix saved to {output_file}")

    print("=== 6. Display and playback generated audio files ===")
    all_files = [dubstep_wav, output_file]
    display_audio_and_links(all_files)
    print("Pipeline complete.")

if __name__ == "__main__":
    main_pipeline()
