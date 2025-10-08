
import numpy as np
from scipy.io.wavfile import write as write_wav

def normalize_audio(audio):
    max_val = np.max(np.abs(audio)) + 1e-9
    if max_val > 1:
        audio = audio / max_val
    return audio.astype(np.float32)

def render_arrangement(arrangement_controller, section_plan, output_path, sample_rate=22050):
    """
    Renders the full audio from an arrangement controller and writes to WAV.

    Args:
        arrangement_controller: Instance managing sequencing and layering.
        section_plan (list): List of (section_name, number_of_steps) tuples.
        output_path (str): Path for saving the WAV.
        sample_rate (int): Audio sample rate.

    Returns:
        None
    """
    print(f"Rendering arrangement and saving to {output_path}...")
    final_audio = arrangement_controller.generate_song(section_plan)
    final_audio = normalize_audio(final_audio)
    write_wav(output_path, sample_rate, (final_audio * 32767).astype(np.int16))
    print(f"Saved rendered arrangement to {output_path}.")
