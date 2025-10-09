# /HybridAI-Music-Composer/voice_module/hil_tts_v2.py

import os
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from sklearn.decomposition import PCA
from IPython.display import Audio, display
import torchaudio
from torchaudio.transforms import MelSpectrogram, GriffinLim
from scipy.fftpack import dct
import scipy

# CRITICAL FIX: Import FlowDenoiser from its dedicated module
from .flow_denoiser import FlowDenoiser # . for relative import within the voice_module package

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_RATE = 16000

# --- Utility functions ---

def load_audio(path, sr=SAMPLE_RATE):
    data, sr0 = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr0 != sr:
        # Resampling logic using torchaudio
        waveform = torch.from_numpy(data).float().unsqueeze(0)
        waveform = torchaudio.transforms.Resample(sr0, sr)(waveform)
        data = waveform.squeeze(0).numpy()
    return data

def save_audio(path, audio, sr=SAMPLE_RATE):
    sf.write(path, audio.astype(np.float32), sr)

def play_audio(audio_or_path, sr=SAMPLE_RATE):
    if isinstance(audio_or_path, str):
        display(Audio(audio_or_path))
    else:
        display(Audio(audio_or_path, rate=sr))

# --- Lightweight Codec wrapper ---

try:
    from encodec import EncodecModel
    enc_model = EncodecModel.encodec_model_48khz()
    enc_model.to(DEVICE)
    enc_model.set_target_bandwidth(6.0)
    ENCODEC_AVAILABLE = True
    print("Encodec available: Using EnCodec model")
except Exception:
    enc_model = None
    ENCODEC_AVAILABLE = False
    print("Encodec not available: Falling back to mel+Griffin-Lim")

_n_mels = 80
_mel_spec = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=_n_mels).to(DEVICE)
_griffin = GriffinLim(n_fft=1024, hop_length=256).to(DEVICE)

def encode_audio_fallback(y_np):
    y = torch.from_numpy(y_np).float().to(DEVICE)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    mel = _mel_spec(y)
    latent = mel.detach().cpu().numpy()[0]
    # The latent embedding is the mean of the mel-bands across time
    emb = np.mean(latent, axis=1) 
    return {'latent': latent, 'embedding': emb}

def decode_audio_fallback(latent):
    # For HIL search, we synthesize audio by tiling the low-dimensional embedding back to a mel-spec
    # Note: latent here is the low-dim embedding (D) from HIL search, not the full mel-spec (D x T)
    D = latent.shape[0]
    T = 80 # Assume a fixed time dimension for the decoder
    mel = torch.from_numpy(latent).float()
    
    # Simple Tiling: Replicate the D-dimensional embedding T times for a (D x T) mel-spec
    # This assumes the low-dim embedding captures the overall spectral shape.
    mel_tiled = mel.unsqueeze(1).repeat(1, T)
    
    # Move to device and add batch dimension
    mel_tiled = mel_tiled.unsqueeze(0).to(DEVICE) 
    wav = _griffin(mel_tiled)
    return wav.squeeze(0).cpu().numpy()

class CodecWrapper:
    def __init__(self, enc_model=None):
        self.enc_model = enc_model
        self.use_encodec = enc_model is not None

    def encode(self, y_np):
        if self.use_encodec:
            # Encodec encoding logic (using codes and averaging for embedding)
            wav = torch.from_numpy(y_np).float().to(DEVICE)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            with torch.no_grad():
                codes, scales, _ = self.enc_model.encode(wav.unsqueeze(0))
            # Average codes across quantization layers and time steps for a single embedding vector
            embedding = codes[0].float().mean(dim=1).mean(dim=1).cpu().numpy()
            return {'codes': codes, 'scales': scales, 'embedding': embedding}
        else:
            return encode_audio_fallback(y_np)

    def decode(self, enc_dict):
        if self.use_encodec:
            # Encodec decoding logic (requires codes and scales)
            codes = enc_dict['codes']
            scales = enc_dict.get('scales')
            
            # Simple HIL: If we only have the *embedding*, we cannot decode via EnCodec directly.
            # We must *reconstruct* the full codes using the FlowDenoiser or similar method.
            # For this simplified demo, we'll assume the full codes are manipulated externally 
            # or the EnCodec decoding is part of a larger flow.
            
            # If we were truly manipulating the embedding, we'd need a model to go from 
            # 'embedding' -> 'codes'. Since the demo skips this, we only decode the original 'codes'.
            
            # If manipulating the embedding, use the fallback method for the demo:
            if 'latent' not in enc_dict and 'embedding' in enc_dict and not 'codes' in enc_dict:
                 return decode_audio_fallback(enc_dict['embedding'])
            
            with torch.no_grad():
                wav = self.enc_model.decode(codes)
            return wav.squeeze(0).cpu().numpy()
        else:
            # Fallback decoding (from Mel)
            return decode_audio_fallback(enc_dict['latent'])

codec = CodecWrapper(enc_model)

# --- Human-in-the-Loop search class ---
class HumanInLoopVoiceSynth:
    """
    Implements the PCA-based search in the latent space (z) guided by a surrogate
    similarity function.
    """
    def __init__(self, initial_embedding, principal_directions, step_sizes, synthesizer_func, max_iterations=32):
        self.z = initial_embedding.copy()
        self.W = principal_directions  # D x N (Principal components)
        self.d = np.array(step_sizes)  # N (Standard deviation along each component)
        self.synth_func = synthesizer_func
        self.max_iter = max_iterations
        self.N = principal_directions.shape[1]
        self.history = []

    def get_candidates(self, iteration):
        """
        Generates 5 candidate points by stepping along the current principal direction.
        The step size decays geometrically with the number of times the direction has been traversed (p).
        """
        n = iteration % self.N # Current dimension index (0 to N-1)
        p = iteration // self.N # Number of full sweeps completed
        step_scale = 2 ** (-p) # Geometric step decay
        
        candidates = []
        for k in [-2, -1, 0, 1, 2]: # Steps: -2d, -1d, 0d, +1d, +2d
            # z_k = current z + k * (decayed step size) * (direction vector)
            z_k = self.z + k * step_scale * self.d[n] * self.W[:, n]
            audio_k = self.synth_func(z_k)
            candidates.append((z_k, audio_k))
        return candidates, n

    def run_search_simulation(self, surrogate_func, ref_audio, max_iter=None):
        """
        Runs the full simulation, selecting the best candidate based on the
        surrogate function's score against the reference audio.
        """
        max_iter = max_iter or self.max_iter
        iteration = 0
        while iteration < max_iter:
            candidates, param_idx = self.get_candidates(iteration)
            audios = [c[1] for c in candidates]
            
            # Surrogate simulates human selection (0 to 4)
            sel = surrogate_func(audios, ref_audio) 
            
            self.z = candidates[sel][0]
            self.history.append((iteration, param_idx, sel, self.z.copy()))
            iteration += 1
            
            if iteration % self.N == 0:
                 print(f"Iteration {iteration}/{max_iter}: Completed sweep {iteration//self.N}")
                 
        return self.z

# --- Surrogate similarity ---

# Resemblyzer fallback
try:
    from resemblyzer import VoiceEncoder
    resey = VoiceEncoder()
    RES_AVAILABLE = True
    print("Resemblyzer available for surrogate similarity")
except ImportError:
    RES_AVAILABLE = False
    print("Resemblyzer not available, using MFCC cosine fallback")

def compute_mfcc_np(y, sr=SAMPLE_RATE, n_mfcc=20):
    # This needs to run on CPU for NumPy/SciPy integration, or use torchaudio MFCC.
    # We use torchaudio/mel-spec setup for consistency then DCT on log-mel to get MFCC.
    y_t = torch.from_numpy(y).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mel = _mel_spec(y_t).cpu().numpy()[0]
    log_mel = np.log(np.maximum(mel, 1e-10))
    # DCT to get cepstral coefficients from log-mel
    mfcc = dct(log_mel, axis=0, type=2, norm='ortho')[:n_mfcc] 
    return np.mean(mfcc, axis=1)

def surrogate_similarity(audios, ref_audio):
    """
    Returns the index of the candidate audio that is most similar to the reference.
    """
    if RES_AVAILABLE:
        # Resemblyzer logic
        ref_emb = resey.embed_utterance(ref_audio) if isinstance(ref_audio, np.ndarray) else resey.embed_file(ref_audio)
        scores = []
        for a in audios:
            emb = resey.embed_utterance(a) 
            sim = np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb) + 1e-12)
            scores.append(sim)
        return int(np.argmax(scores))
    else:
        # MFCC Cosine Similarity Fallback
        ref_m = compute_mfcc_np(ref_audio) if isinstance(ref_audio, np.ndarray) else compute_mfcc_np(load_audio(ref_audio))
        scores = []
        for a in audios:
            am = compute_mfcc_np(a)
            sim = np.dot(ref_m, am) / (np.linalg.norm(ref_m) * np.linalg.norm(am) + 1e-12)
            scores.append(sim)
        return int(np.argmax(scores))

# --- Demo runner ---
def demo_simulation(reference_path, target_path=None, max_iter=16):
    """
    Runs a full HIL simulation: encodes audio, performs PCA on latent space,
    searches for a better latent point, and synthesizes the result.
    """
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference audio not found at: {reference_path}")
        
    ref_y = load_audio(reference_path)
    enc_ref = codec.encode(ref_y)
    ref_emb = enc_ref['embedding']

    D = ref_emb.shape[0] # Latent dimension size
    num_samples = max(64, 2 * D) # Dataset size for PCA
    rng = np.random.RandomState(42)
    
    # Create mock latent data by adding noise to the reference embedding
    emb_matrix = np.stack([ref_emb + 0.5 * rng.normal(size=D) for _ in range(num_samples)])

    N = min(16, D) # Number of principal components to search along
    pca = PCA(n_components=N)
    pca.fit(emb_matrix)
    W = pca.components_.T # Principal directions (D x N)
    projected = pca.transform(emb_matrix)
    d = np.std(projected, axis=0) + 1e-6 # Step size scale along each component

    def synthesizer_from_alpha(alpha_vec):
        """Mock synthesizer function that decodes the HIL-modified embedding."""
        
        # NOTE: This is the crucial simplification for the demo. 
        # In a real flow model, alpha_vec (the embedding) would be used to condition 
        # the FlowDenoiser to generate the full latent codes/mels.
        
        if ENCODEC_AVAILABLE:
            # Full codec approach requires mapping alpha_vec to codes, which is complex.
            # We stick to the simplified fallback for consistent HIL search.
            print("WARNING: Using simplified mel/Griffin-Lim fallback for HIL decoding.")
            pass # Fall through to fallback
            
        # Simplified Fallback for both Codec and Mel:
        # The embedding (alpha_vec) is used as the base for the decoder.
        # We pass the new embedding in a dict format the CodecWrapper can handle.
        enc = {'latent': alpha_vec, 'embedding': alpha_vec}
        return codec.decode(enc)

    z0 = pca.mean_ # Start search from the mean of the perturbed space
    hil = HumanInLoopVoiceSynth(z0, W, d, synthesizer_from_alpha, max_iterations=max_iter)
    
    # If target_path is provided, the goal is to match that sound; otherwise, match ref_y
    ref_for_surrogate = load_audio(target_path) if target_path else ref_y
    
    print("Starting simulation...")
    final_z = hil.run_search_simulation(surrogate_similarity, ref_for_surrogate, max_iter=max_iter)
    final_audio = synthesizer_from_alpha(final_z)
    save_audio('hil_final.wav', final_audio)
    print(f"Final audio saved: hil_final.wav (search goal: {'match target' if target_path else 'denoise self'})")
    play_audio(final_audio)
    return hil

print("\n--- HIL-TTS v2 module loaded ---")
print("Use `demo_simulation(reference_path, target_path=None, max_iter=16)` to run demo.")
