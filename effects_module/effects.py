# /main_pyo_synth.py (FINAL PYO STABILITY FIX - TESTING DRY SIGNAL ONLY)

import time
import numpy as np
from pyo import *
import threading
import queue
import mido
import os
# NOTE: Ensure 'pyo_effects.granular' is correctly imported or mocked if not in the path
try:
    from pyo_effects.granular import GranularEffect 
except ImportError:
    class GranularEffect:
        def __init__(self, *args, **kwargs): pass
        def get_output(self): return Sig(0) # Mock silent output
    print("[Warning] Could not import GranularEffect. Using silent mock object.")
    
# --- CONFIGURATION (Match previous successful output rate for stability) ---
CC_GRAN_PITCH = 20
CC_GRAN_MIX = 21

# Audio params
SAMPLE_RATE = 22050  # <-- CRITICAL FIX: Match previous successful WAV output
BLOCK_SIZE = 512     # Default block size
CHANNELS = 2
RENDER_DURATION_SECONDS = 16

# --- PYO SERVER SETUP (OFFLINE MODE) ---

# CRITICAL FIX: Explicitly set the internal buffer size *before* booting the Server.
pyo.pa_set_buffer_size(BLOCK_SIZE)

# 1. Initialize Server: Use the forced sample rate.
s = Server(sr=SAMPLE_RATE, nchnls=CHANNELS, buffersize=BLOCK_SIZE)

# 2. Configure for OFFLINE (non-realtime) MODE.
s.setInOut(0, 0)
s.setHostAudio(0, 0)
s.boot() 
# NOTE: We DO NOT call s.start() here.
# -------------------------------------------------------------------------


# --- SYNTHESIS CLASSES (Your Code) ---

class Oscillator:
    def __init__(self):
        # Using SawDPW is fine, but if static persists after this, try SawTable.
        self.pyo_osc = SawDPW(freq=440, mul=0)
        self.pyo_filt = MoogLP(self.pyo_osc, freq=1000, res=0.5)
        self.pyo_env = Adsr(attack=0.01, decay=0.3, sustain=0.0, release=0.1, dur=0.5, mul=1)
        self.pyo_osc.mul = self.pyo_env 
    def set_freq(self, freq): self.pyo_osc.freq = freq
    def set_cutoff(self, val): self.pyo_filt.freq = 200 + val / 127.0 * 10000
    def set_reso(self, val): self.pyo_filt.res = val / 127.0 * 5.0
    def note_on(self, vel): self.pyo_env.play()
    def note_off(self, vel): self.pyo_env.stop() 
    def get_output(self): return self.pyo_filt

class DrumKit:
    def __init__(self):
        # SfPlayer will use the server's SAMPLE_RATE (22050 Hz)
        self.players = {36: SfPlayer("samples/bd01.wav", mul=0)} 
    def trigger(self, note, vel): 
        if note in self.players: self.players[note].out()
    def get_output(self): 
        return Sum(list(self.players.values()))

# --- GLOBAL INSTANCES ---

osc1 = Oscillator()
osc2 = Oscillator()
drums = DrumKit()

# Granular effect is mocked or initialized but BYPASSED for this test
GRANULAR_SOURCE = "dubstep_wobble_output.wav" 
granular_effect = GranularEffect(input_source=GRANULAR_SOURCE) 


# --- PYO SIGNAL CHAIN (ISOLATION TEST) ---

# 1. Mixer: Sum all dry signals
dry_mix = Sum([osc1.get_output(), osc2.get_output(), drums.get_output()])

# 2. Safety Scaling: Crucial to prevent digital clipping (which sounds like static)
final_dry_signal = dry_mix * 0.9

# --- FINAL PYOO OUTPUT OBJECT (Connect Dry Signal Directly) ---
final_pyo_output_object = final_dry_signal # <-- The final signal to be recorded

# NOTE: MIDI logic is bypassed for this file generation script.

# --- OFFLINE RENDERING FUNCTION ---

def render_and_save_audio(pyo_object_to_record, duration_seconds, output_path="dubstep_wobble_output.wav"):
    """
    Renders the Pyo signal directly to a WAV file and then stops the server.
    """
    # 1. Start the server (required before recording/rendering begins)
    s.start() 
    
    # 2. Tell Pyo what to record
    s.rec(
        pyo_object_to_record,  # The final signal chain (now the dry mix)
        filename=output_path,  # The target file path
        dur=duration_seconds   # The total duration to render
    )
    
    print(f"Pyo is rendering {duration_seconds} seconds to {output_path} at {s.getSr()} Hz...")
    
    # 3. Stop the server after rendering is done
    s.stop() 
    
    # Wait to ensure the process completes
    time.sleep(duration_seconds + 1) 

    print("Rendering complete. Server stopped.")
    return output_path

# --- SCRIPT EXECUTION HOOK ---
# The main `generate_dubstep` script must now call this function 
# using the final_pyo_output_object.

# Example of how the main script should call it (if the logic is external):
# output_file = render_and_save_audio(final_pyo_output_object, RENDER_DURATION_SECONDS)
