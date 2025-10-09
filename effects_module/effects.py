# /main_pyo_synth.py (UPDATED for stable OFFLINE RENDERING in Colab)

import time
import numpy as np
from pyo import *
import threading
import queue
import mido
import os
# Import GranularEffect (ensure pyo_effects/granular.py is imported correctly)
# from pyo_effects.granular import GranularEffect 
# ... other imports ...

# Assuming midi_config is correct
CC_GRAN_PITCH = 20
CC_GRAN_MIX = 21

# --- AUDIO PARAMETERS (44.1 kHz Standard) ---
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
CHANNELS = 2
RENDER_DURATION_SECONDS = 16  # Match your typical output length

# --- PYOO OFFLINE RENDERING SETUP ---
# 1. Initialize Server: Use standard parameters.
s = Server(sr=SAMPLE_RATE, nchnls=CHANNELS, buffersize=BLOCK_SIZE)

# 2. Configure for OFFLINE (non-realtime) MODE.
# This prevents real-time I/O conflicts common in remote environments like Colab.
# Set I/O to zero to force file rendering.
s.setInOut(0, 0)
s.setHostAudio(0, 0)
s.boot() # Boot the server immediately after configuration

# Note: We DON'T call s.start() here.
# ---------------------------------------------


# --- Placeholder/Mock Classes (replace with your full pyo-based classes) ---
class Oscillator:
    def __init__(self):
        # Using Pyo objects directly for efficiency
        self.pyo_osc = SawDPW(freq=440, mul=0)
        self.pyo_filt = MoogLP(self.pyo_osc, freq=1000, res=0.5)
        self.pyo_env = Adsr(attack=0.01, decay=0.3, sustain=0.0, release=0.1, dur=0.5, mul=1)
        self.pyo_osc.mul = self.pyo_env 
    def set_freq(self, freq): self.pyo_osc.freq = freq
    def set_cutoff(self, val): self.pyo_filt.freq = 200 + val / 127.0 * 10000
    def set_reso(self, val): self.pyo_filt.res = val / 127.0 * 5.0
    def note_on(self, vel): self.pyo_env.play()
    def note_off(self, vel): self.pyo_env.stop() # Added vel argument for consistency if needed
    def get_output(self): return self.pyo_filt

class DrumKit:
    def __init__(self):
        # Ensure your sample rate is consistent with Pyo server (44100 Hz)
        self.players = {36: SfPlayer("samples/bd01.wav", mul=0)} 
    def trigger(self, note, vel): 
        if note in self.players: self.players[note].out()
    def get_output(self): 
        return Sum(list(self.players.values()))

# Assuming the GranularEffect class from section 1 is available
# NOTE: Ensure "dubstep_wobble_output.wav" is generated before this runs,
# or use a placeholder file for testing the Pyo chain.
GRANULAR_SOURCE = "dubstep_wobble_output.wav" 
# Ensure GranularEffect is imported and defined correctly.
# granular_effect = GranularEffect(input_source=GRANULAR_SOURCE) 
# --- End Mocks ---

# Global instances (You need to uncomment and define your actual classes here)
# osc1 = Oscillator()
# osc2 = Oscillator()
# drums = DrumKit()
# granular_effect = GranularEffect(input_source=GRANULAR_SOURCE) 


# Global Pyo effects chain 
# 1. Mixer: Sum all dry signals
# dry_mix = Sum([osc1.get_output(), osc2.get_output(), drums.get_output()])

# 2. Granular: Treat the granular effect as a send/insert, mixing wet/dry later
# granular_wet = granular_effect.get_output()

# 3. Main Effects (Delay/Reverb/Compressor)
# For this example, we'll mock the final output object to enable rendering
final_pyo_output_object = Sig(Sine(freq=440, mul=0.5) * 0.5) # Mock final output

# final_out = compressor.out() # This is the object Pyo needs to record

midi_queue = queue.Queue()
# ... (midi_listener function, process_midi thread setup, etc.) ...

# --- OFFLINE RENDERING FUNCTION ---

def render_and_save_audio(pyo_object_to_record, duration_seconds, output_path="dubstep_wobble_output.wav"):
    """
    Renders the Pyo signal directly to a WAV file and then stops the server.
    This replaces the continuous s.gui() or s.start() loop for file generation.
    """
    # 1. Start the server (required before recording/rendering begins)
    s.start() 
    
    # 2. Tell Pyo what to record
    s.rec(
        pyo_object_to_record,  # The Pyo object holding the final signal chain
        filename=output_path,  # The target file path
        dur=duration_seconds   # The total duration to render
    )
    
    print(f"Pyo is rendering {duration_seconds} seconds to {output_path} at {s.getSr()} Hz...")
    
    # 3. Wait for the rendering process to complete (pyo does this internally)
    # Since we replaced the real-time loop, the script should exit cleanly after rec.
    
    # 4. Stop the server after rendering is done
    s.stop() 
    
    # You might need to add a small sleep or check if the file exists before stopping
    # depending on how pyo handles the completion in non-real-time mode.
    # A simple way to wait:
    time.sleep(duration_seconds + 1) # Wait slightly longer than the duration

    print("Rendering complete. Server stopped.")
    
    # NOTE: Your main script will call this function instead of the midi/pyo loop.
    return output_path

# Example of how the main `generate_dubstep` script would use this function:
# output_file = render_and_save_audio(final_pyo_output_object, RENDER_DURATION_SECONDS)
