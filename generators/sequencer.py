
import numpy as np

class StepSequencer:
    def __init__(self, layers, pattern_length):
        """
        Args:
            layers (list): List of AudioLayer instances.
            pattern_length (int): Number of steps in the repeating pattern.
        """
        self.layers = layers
        self.pattern_length = pattern_length
        self.current_step = 0

    def synth_next_step(self):
        """
        Synthesize one step for all layers, mix them, and advance the step counter.

        Returns:
            np.ndarray: Stereo audio buffer for this step, shape (samples_per_step, 2).
        """
        stereo_buffer = np.zeros((samples_per_step, 2), dtype=np.float32)
        for layer in self.layers:
            step_audio = layer.get_precomputed_step(self.current_step)
            if step_audio is not None:
                stereo_buffer += step_audio
        self.current_step = (self.current_step + 1) % self.pattern_length
        # Clamp to [-1,1] to avoid clipping
        stereo_buffer = np.clip(stereo_buffer, -1.0, 1.0)
        return stereo_buffer

class ArrangementController:
    def __init__(self, sequencer, total_steps):
        """
        Args:
            sequencer (StepSequencer): The sequencer managing all layers.
            total_steps (int): Total number of steps to generate the full song.
        """
        self.sequencer = sequencer
        self.total_steps = total_steps
        self.current_step = 0

    def generate_song(self, section_plan):
        """
        Generate the entire song based on provided section plan.

        Args:
            section_plan (list of tuples): Each tuple is (section_name, num_steps).

        Returns:
            np.ndarray: Full stereo audio waveform concatenated.
        """
        full_audio = []

        for section_name, num_steps in section_plan:
            print(f"Generating section '{section_name}' with {num_steps} steps...")
            for step_in_section in range(num_steps):
                step_audio = self.sequencer.synth_next_step()
                self.current_step += 1

                # Example automation: volume ramp in build-up sections
                if 'build-up' in section_name.lower():
                    volume_scale = 0.5 + (step_in_section / num_steps) * 0.5  # ramp 0.5->1.0
                    step_audio *= volume_scale

                full_audio.append(step_audio)

        song = np.concatenate(full_audio, axis=0)
        # Normalize final mix
        max_val = np.max(np.abs(song)) + 1e-9
        return (song / max_val).astype(np.float32)
