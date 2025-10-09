# /pyo_effects/granular.py (New file)

from pyo import *
import numpy as np

# Pyo Server (assuming the global server 's' is already booted and started)
# s = Server(sr=44100, nchnls=2, buffersize=512).boot().start() 

class GranularEffect:
    """
    Pyo-based Granular Synthesizer wrapper, controllable via MIDI CC.
    Can operate on a loaded sound file or an input signal.
    """
    def __init__(self, input_source=None, sr=44100, buffer_size=4096):
        self.sr = sr
        self.buffer_size = buffer_size
        self.table = None
        self.granulator = None
        self.env = HannTable() # Smooth Hanning window
        
        # Default modulation parameters
        self.pitch_factor = 1.0
        self.grain_duration = 0.1
        self.grain_count = 16
        self.mix = 0.0 # Wet/Dry Mix
        
        # Pyo objects for real-time control (will be initialized later)
        self.cc_pitch = Sig(self.pitch_factor)
        self.cc_dur = Sig(self.grain_duration)
        self.cc_mix = Sig(self.mix)

        self.load_input(input_source)


    def load_input(self, input_source):
        """Loads a sound file or sets up for live input."""
        if isinstance(input_source, str) and os.path.exists(input_source):
            # Load a sound file into a PyoTable
            self.table = SndTable(input_source)
            # Position: Scanning across the table with a slight random jitter
            pos_phasor = Phasor(freq=self.table.getRate() * 0.1) # Scan speed 10%
            self.pos = pos_phasor + Noise(mul=0.002, add=0)
            print(f"[Granular] Loaded file: {input_source}")
            
        elif input_source is None:
            # Placeholder for live input/audio chain input
            print("[Granular] Initializing for live input/chain processing.")
            self.table = Input() # Use Input() as the source
            # Set a simple, randomized position within the current buffer
            self.pos = Noise(mul=0.05, add=0.5) 
            
        self._init_granulator()


    def _init_granulator(self):
        """Initializes or re-initializes the Pyo Granulator object."""
        if self.granulator:
            self.granulator.stop()
        
        if self.table is None:
            # If no input, use silence as a placeholder table
            self.table = SineTable() 

        # Duration modulation (small jitter around the CC-controlled duration)
        dur_mod = self.cc_dur + Noise(mul=0.005)

        # The core Granulator setup
        self.granulator = Granulator(
            table=self.table, 
            env=self.env, 
            pitch=self.cc_pitch, 
            pos=self.pos, 
            dur=dur_mod, 
            grains=self.grain_count, 
            basedur=0.1, 
            mul=0.5 # Default amplitude scaling
        )
        
        # Wet/Dry mix: Mix the granular output with silence (or the dry input if chaining)
        # Note: In an effects chain, you'd typically mix the wet signal (self.granulator)
        # with the dry signal coming from the rest of the synth.
        self.output = Sig(self.granulator, mul=self.cc_mix)
        # self.output.out() # Don't .out() here, let the main server chain handle it


    def set_pitch(self, val):
        """CC for pitch (e.g., CC 17) -> Controls grain pitch multiplier."""
        # Map CC (0-127) to a useful pitch range (e.g., 0.5x to 2.0x)
        pitch_factor = 0.5 + (val / 127.0) * 1.5
        self.cc_pitch.value = pitch_factor
        self.pitch_factor = pitch_factor

    def set_duration(self, val):
        """CC for duration (e.g., CC 18) -> Controls grain duration."""
        # Map CC (0-127) to 20ms - 200ms
        dur = 0.02 + (val / 127.0) * 0.18 
        self.cc_dur.value = dur
        self.grain_duration = dur

    def set_mix(self, val):
        """CC for wet/dry mix (e.g., CC 19)."""
        mix = val / 127.0
        self.cc_mix.value = mix
        self.mix = mix

    def get_output(self):
        """Returns the Pyo output signal for chaining."""
        return self.output
