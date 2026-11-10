"""Shared helpers for Lite.py and Pro.py.

Audio is held as float32 arrays of shape (n_samples, n_channels) throughout.
Heavy dependencies (soundfile, librosa, torch, FlashSR) are imported lazily so the
DSP helpers can be imported and tested without them.
"""
from pathlib import Path

import numpy as np
from scipy.signal.windows import hann

SR = 48000
WIN = 245760          # FlashSR's fixed input length (5.12 s @ 48 kHz)
HOP = WIN // 2        # 50 % overlap with a periodic Hann window sums to exactly 1


def load_audio(path):
    """Return (audio[n, ch] float32, sample_rate). >2 channels are downmixed to mono."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] > 2:
        keep = list(range(audio.shape[1]))
        if audio.shape[1] == 6:   # assume standard 5.1 order FL FR FC LFE BL BR
            keep.remove(3)
            print("  [warn] 6 channels: assuming 5.1 order, excluding LFE (ch 4), downmixing to mono")
        else:
            print(f"  [warn] {audio.shape[1]} channels found; downmixing all to mono")
        audio = audio[:, keep].mean(axis=1, keepdims=True)
    return audio, sr


def resample(audio, sr, target=SR):
    """Resample along the TIME axis (axis=0) of an (n, ch) array."""
    if sr == target:
        return audio
    import librosa
    out = librosa.resample(audio, orig_sr=sr, target_sr=target, axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def chunked_apply(x, fn, win=WIN, hop=HOP):
    """Apply a fixed-length model `fn` (win samples in -> win samples out) to a 1-D
    signal of any length using Hann-windowed overlap-add.

    The signal is padded by `hop` on both sides (reflect when longer than `hop`, else
    zeros) so every real sample is covered by windows summing to 1. Pad samples are
    cropped from the output, but `fn` still sees the padding and its output on chunks
    that contain pad can influence the edge of the real signal. Output length == input.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    L = len(x)
    if L == 0:
        return x.copy()
    xp = np.pad(x, (hop, hop), mode="reflect" if L > hop else "constant")
    P = len(xp)
    n = max(1, -(-(P - win) // hop) + 1)
    total = (n - 1) * hop + win
    if total > P:
        xp = np.pad(xp, (0, total - P))
    w = hann(win, sym=False).astype(np.float32)
    out = np.zeros(total, np.float32)
    norm = np.zeros(total, np.float32)
    for i in range(n):
        s = i * hop
        y = np.asarray(fn(xp[s:s + win]), dtype=np.float32).reshape(-1)
        if y.shape[0] != win:
            raise ValueError(f"model returned {y.shape[0]} samples for a {win}-sample window")
        out[s:s + win] += y * w
        norm[s:s + win] += w
    out /= np.maximum(norm, 1e-6)
    return out[hop:hop + L]


def load_flashsr(weights_dir, device):
    try:
        from FlashSR.FlashSR import FlashSR
    except ImportError as e:
        raise ImportError("FlashSR is not installed; see README (Installation)") from e
    w = Path(weights_dir)
    # Positional args follow the FlashSR_Inference README example.
    model = FlashSR(str(w / "student_ldm.pth"), str(w / "sr_vocoder.pth"), str(w / "vae.pth"))
    return model.to(device).eval()


def super_resolve(audio, model, device):
    """FlashSR on every channel independently (each pass is mono, shape (1, WIN))."""
    import torch

    def fn(chunk):
        t = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t, lowpass_input=False)
        return out.detach().float().cpu().numpy().reshape(-1)

    return np.column_stack([chunked_apply(audio[:, c], fn) for c in range(audio.shape[1])])


def loudness_normalize(audio, target_lufs):
    try:
        import pyloudnorm as pyln
    except ImportError:
        print("  [warn] pyloudnorm not installed; skipping loudness normalisation")
        return audio
    try:
        meter = pyln.Meter(SR)
        loud = meter.integrated_loudness(audio if audio.shape[1] > 1 else audio[:, 0])
        if np.isfinite(loud):
            audio = pyln.normalize.loudness(audio, loud, target_lufs)
    except ValueError as e:
        print(f"  [warn] loudness normalisation skipped: {e}")
    return audio


def peak_guard(audio, ceiling=0.989):
    """Scale down (never up) so the peak stays below ~-0.1 dBFS; warns when it engages."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > ceiling:
        db = 20 * np.log10(ceiling / peak)
        print(f"  [warn] peak {peak:.3f} > {ceiling}; scaled by {db:.2f} dB to avoid clipping "
              f"(any LUFS target is now missed by about that amount)")
        return audio * (ceiling / peak)
    return audio


def save_audio(path, audio, subtype="PCM_24"):
    import soundfile as sf
    audio = peak_guard(audio)
    sf.write(str(path), audio if audio.shape[1] > 1 else audio[:, 0], SR, subtype=subtype)
