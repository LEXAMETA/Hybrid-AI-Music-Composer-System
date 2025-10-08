# /HybridAI-Music-Composer/scripts/train_adapt.py

import argparse
import time
# Assuming the corrected imports:
from audio_layers.audio_layer import AudioLayer
# Mock style_transfer import as it wasn't provided in full:
class AudioLayerExt(AudioLayer): pass
def style_transfer_train(wav_path, base_layer, epochs):
    print(f"MOCK: Training {base_layer.name} adaptation on {wav_path} for {epochs} epochs.")
    # In a real system, this would call AudioLayerExt methods
    return AudioLayerExt(base_layer)

def main(wav_path, epochs):
    start_time = time.time()
    print(f"Loading and training adaptation from external WAV: {wav_path}")
    
    # CRITICAL: In a real system, base_layer must be trained first to initialize the model/scalers!
    base_layer = AudioLayer(base_freq=60.0, pan=0.0, name="base_layer")
    # base_layer.train_model(epochs=30) # Real training call here
    
    adapted_layer = style_transfer_train(wav_path, base_layer, epochs=epochs)
    print(f"Adaptation training complete. Time elapsed: {time.time() - start_time:.2f} seconds")

    # Optionally save model weights for reuse
    weights_path = "adapted_audio_layer_weights.h5"
    # adapted_layer.model.save(weights_path) # Real save call here
    print(f"Saved adapted model weights (MOCK) to: {weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style transfer training runner.")
    parser.add_argument("--wav", type=str, required=True, help="Path to external WAV file for adaptation.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")

    args = parser.parse_args()
    main(args.wav, args.epochs)
