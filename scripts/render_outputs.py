# /HybridAI-Music-Composer/scripts/render_outputs.py

import numpy as np
from scipy.io.wavfile import write as write_wav

def normalize_audio(audio):
    """Normalizes floating-point audio to range [-1, 1]."""
    max_val = np.max(np.abs(audio)) + 1e-9
    # Only scale down if necessary, audio should already be mostly normalized
    if max_val > 1.0: 
        audio = audio / max_val
    return audio.astype(np.float32)

def render_arrangement(arrangement_controller, section_plan, output_path, sample_rate=22050):
    """
    Renders the full audio from an arrangement controller and writes to WAV.
    """
    print(f"Rendering arrangement and saving to {output_path}...")
    
    # Call the core generation method
    final_audio = arrangement_controller.generate_song(section_plan)
    
    # Final normalization before writing to PCM
    final_audio = normalize_audio(final_audio)
    
    # Save as 16-bit PCM
    write_wav(output_path, sample_rate, (final_audio * 32767).astype(np.int16))
    print(f"Saved rendered arrangement to {output_path}.")
