
# /main_pyo_synth.py (Adapted from your previous code)

import time
import numpy as np
from pyo import *
import threading
import queue
import mido
import os
# ... other imports ...

# Assuming midi_config is correct
CC_GRAN_PITCH = 20
CC_GRAN_MIX = 21

# Audio params
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
CHANNELS = 2

# Pyo Server integration
s = Server(sr=SAMPLE_RATE, nchnls=CHANNELS, buffersize=BLOCK_SIZE).boot()
s.start()

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
    def note_off(self): self.pyo_env.stop()
    def get_output(self): return self.pyo_filt

class DrumKit:
    def __init__(self):
        self.players = {36: SfPlayer("samples/bd01.wav", mul=0)} # Mock
    def trigger(self, note, vel): 
        if note in self.players: self.players[note].out()
    def get_output(self): 
        # In a real pyo system, you'd mix the drum players here
        return Sum(list(self.players.values()))

# Assuming the GranularEffect class from section 1 is available
# from pyo_effects.granular import GranularEffect 
# --- End Mocks ---

# Global instances
osc1 = Oscillator()
osc2 = Oscillator()
drums = DrumKit()

# Initialize Granular Effect: Use a placeholder file for testing
# NOTE: The style-transferred audio (dubstep_wobble_output.wav) should be loaded here
GRANULAR_SOURCE = "dubstep_wobble_output.wav" # Ensure this file exists
granular_effect = GranularEffect(input_source=GRANULAR_SOURCE) 

# Global Pyo effects chain (Order matters!)
# 1. Mixer: Sum all dry signals
dry_mix = Sum([osc1.get_output(), osc2.get_output(), drums.get_output()])

# 2. Granular: Treat the granular effect as a send/insert, mixing wet/dry later
granular_wet = granular_effect.get_output()

# 3. Main Effects (Delay/Reverb/Compressor)
delay_effect = Delay(dry_mix + granular_wet, delay=0.5, feedback=0.5) # Apply delay to the granular output too
reverb = Freeverb(delay_effect, size=0.5, damp=0.5)
compressor = Compress(reverb, thresh=-20, ratio=4)

# Output the final mix
final_out = compressor.out() # This line sends the whole chain to the audio output

midi_queue = queue.Queue()

# MIDI Listener (same as before)
# ... (midi_listener function here) ...

# MIDI Processing Thread
def process_midi():
    while s.getIsRunning():
        if not midi_queue.empty():
            msg = midi_queue.get()
            if msg.type == 'note_on' or msg.type == 'note_off':
                # Handle notes for osc1, osc2, drums
                pass # Logic here...
            elif msg.type == 'control_change':
                val = msg.value
                
                # --- Granular Control ---
                if msg.control == CC_GRAN_PITCH:
                    granular_effect.set_pitch(val)
                elif msg.control == CC_GRAN_MIX:
                    granular_effect.set_mix(val)
                # --- End Granular Control ---
                
                # ... other CC controls (cutoff, reso, etc.) ...
        time.sleep(0.001)

process_thread = threading.Thread(target=process_midi)
process_thread.start()

# Run server loop
try:
    print('Pyo Synth running with Granular Effect. Press Ctrl+C to stop.')
    s.gui(locals()) # Use Pyo GUI for real-time control/monitoring
except KeyboardInterrupt:
    s.stop()
    process_thread.join()
