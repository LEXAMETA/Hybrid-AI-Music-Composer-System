
import argparse
import time
from audio_layers.audio_layer import AudioLayer
from style_transfer.style_adapt import style_transfer_train

def main(wav_path, epochs):
    start_time = time.time()
    print(f"Loading and training adaptation from external WAV: {wav_path}")
    base_layer = AudioLayer(base_freq=60.0, pan=0.0, name="base_layer")
    adapted_layer = style_transfer_train(wav_path, base_layer, epochs=epochs)
    print(f"Adaptation training complete. Time elapsed: {time.time() - start_time:.2f} seconds")

    # Optionally save model weights for reuse
    weights_path = "adapted_audio_layer_weights.h5"
    adapted_layer.model.save(weights_path)
    print(f"Saved adapted model weights to: {weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style transfer training runner.")
    parser.add_argument("--wav", type=str, required=True, help="Path to external WAV file for adaptation.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")

    args = parser.parse_args()
    main(args.wav, args.epochs)
