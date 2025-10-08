
import os
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from sklearn.decomposition import PCA
from IPython.display import Audio, display

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_RATE = 16000

# Utility functions

def load_audio(path, sr=SAMPLE_RATE):
    data, sr0 = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr0 != sr:
        import torchaudio
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

# Lightweight Codec wrapper (try EnCodec, fallback mel+Griffin-Lim)

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

import torchaudio
from torchaudio.transforms import MelSpectrogram, GriffinLim

_n_mels = 80
_mel_spec = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=_n_mels).to(DEVICE)
_griffin = GriffinLim(n_fft=1024, hop_length=256).to(DEVICE)

def encode_audio_fallback(y_np):
    y = torch.from_numpy(y_np).float().to(DEVICE)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    mel = _mel_spec(y)
    latent = mel.detach().cpu().numpy()[0]
    emb = np.mean(latent, axis=1)
    return {'latent': latent, 'embedding': emb}

def decode_audio_fallback(latent):
    mel = torch.from_numpy(latent).unsqueeze(0).to(DEVICE)
    wav = _griffin(mel)
    return wav.squeeze(0).cpu().numpy()

class CodecWrapper:
    def __init__(self, enc_model=None):
        self.enc_model = enc_model
        self.use_encodec = enc_model is not None

    def encode(self, y_np):
        if self.use_encodec:
            wav = torch.from_numpy(y_np).float().to(DEVICE)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            with torch.no_grad():
                codes = self.enc_model.encode(wav)
            embedding = torch.mean(codes[0].float(), dim=1).mean(dim=1).cpu().numpy()
            return {'codes': codes, 'embedding': embedding}
        else:
            return encode_audio_fallback(y_np)

    def decode(self, enc_dict):
        if self.use_encodec:
            codes = enc_dict['codes']
            with torch.no_grad():
                wav = self.enc_model.decode(codes)
            return wav.squeeze(0).cpu().numpy()
        else:
            return decode_audio_fallback(enc_dict['latent'])

codec = CodecWrapper(enc_model)

# Flow Denoiser - Toy continuous latent generator

class FlowDenoiser(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x_t, t):
        t = t.view(-1, 1)
        inp = torch.cat([x_t, t], dim=1)
        return self.net(inp)

# Human-in-the-Loop search class
class HumanInLoopVoiceSynth:
    def __init__(self, initial_embedding, principal_directions, step_sizes, synthesizer_func, max_iterations=32):
        self.z = initial_embedding.copy()
        self.W = principal_directions  # D x N
        self.d = np.array(step_sizes)
        self.synth_func = synthesizer_func
        self.max_iter = max_iterations
        self.N = principal_directions.shape[1]
        self.history = []

    def get_candidates(self, iteration):
        n = iteration % self.N
        p = iteration // self.N
        step_scale = 2 ** (-p)
        candidates = []
        for k in [-2, -1, 0, 1, 2]:
            z_k = self.z + k * step_scale * self.d[n] * self.W[:, n]
            audio_k = self.synth_func(z_k)
            candidates.append((z_k, audio_k))
        return candidates, n

    def run_search_simulation(self, surrogate_func, ref_audio, max_iter=None):
        max_iter = max_iter or self.max_iter
        iteration = 0
        while iteration < max_iter:
            candidates, param_idx = self.get_candidates(iteration)
            audios = [c[1] for c in candidates]
            sel = surrogate_func(audios, ref_audio)
            self.z = candidates[sel][0]
            self.history.append((iteration, param_idx, sel, self.z.copy()))
            iteration += 1
        return self.z

# Surrogate similarity - Resemblyzer fallback
try:
    from resemblyzer import VoiceEncoder
    resey = VoiceEncoder()
    RES_AVAILABLE = True
    print("Resemblyzer available for surrogate similarity")
except ImportError:
    RES_AVAILABLE = False
    print("Resemblyzer not available, using MFCC cosine fallback")

import scipy
from scipy.fftpack import dct

def compute_mfcc_np(y, sr=SAMPLE_RATE, n_mfcc=20):
    y_t = torch.from_numpy(y).float().unsqueeze(0).to(DEVICE)
    mel = _mel_spec(y_t).cpu().numpy()[0]
    log_mel = np.log(np.maximum(mel, 1e-10))
    mfcc = dct(log_mel, axis=0, type=2, norm='ortho')[:n_mfcc]
    return np.mean(mfcc, axis=1)

def surrogate_similarity(audios, ref_audio):
    if RES_AVAILABLE:
        ref_emb = resey.embed_utterance(ref_audio) if isinstance(ref_audio, np.ndarray) else resey.embed_file(ref_audio)
        scores = []
        for a in audios:
            emb = resey.embed_utterance(a) if isinstance(a, np.ndarray) else resey.embed_file(a)
            sim = np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb) + 1e-12)
            scores.append(sim)
        return int(np.argmax(scores))
    else:
        ref_m = compute_mfcc_np(ref_audio) if isinstance(ref_audio, np.ndarray) else compute_mfcc_np(load_audio(ref_audio))
        scores = []
        for a in audios:
            am = compute_mfcc_np(a) if isinstance(a, np.ndarray) else compute_mfcc_np(load_audio(a))
            sim = np.dot(ref_m, am) / (np.linalg.norm(ref_m) * np.linalg.norm(am) + 1e-12)
            scores.append(sim)
        return int(np.argmax(scores))

# Demo runner
def demo_simulation(reference_path, target_path=None, max_iter=16):
    ref_y = load_audio(reference_path)
    enc_ref = codec.encode(ref_y)
    ref_emb = enc_ref['embedding']

    D = ref_emb.shape[0]
    num_samples = max(64, 2 * D)
    rng = np.random.RandomState(42)
    emb_matrix = np.stack([ref_emb + 0.5 * rng.normal(size=D) for _ in range(num_samples)])

    N = min(16, D)
    pca = PCA(n_components=N)
    pca.fit(emb_matrix)
    W = pca.components_.T
    projected = pca.transform(emb_matrix)
    d = np.std(projected, axis=0) + 1e-6

    def synthesizer_from_alpha(alpha_vec):
        if ENCODEC_AVAILABLE:
            enc = {'codes': enc_ref['codes'], 'embedding': alpha_vec}
            return codec.decode(enc)
        else:
            latent = np.tile(alpha_vec[:, None], (1, 80))
            return decode_audio_fallback(latent)

    z0 = pca.mean_
    hil = HumanInLoopVoiceSynth(z0, W, d, synthesizer_from_alpha, max_iterations=max_iter)
    ref_for_surrogate = load_audio(target_path) if target_path else ref_y
    
    print("Starting simulation...")
    final_z = hil.run_search_simulation(surrogate_similarity, ref_for_surrogate, max_iter=max_iter)
    final_audio = synthesizer_from_alpha(final_z)
    save_audio('hil_final.wav', final_audio)
    print("Final audio saved: hil_final.wav")
    play_audio(final_audio)
    return hil

print("
--- HIL-TTS v2 module loaded ---")
print("Use `demo_simulation(reference_path, target_path=None, max_iter=16)` to run demo.")
