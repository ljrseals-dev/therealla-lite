"""TheRealla Lite: resample -> FlashSR -> optional loudness normalisation.

Mono in -> mono out. Stereo in -> stereo out (each channel is enhanced independently).
Audio of any length is handled by chunked overlap-add (FlashSR itself is fixed at 5.12 s).
"""
import argparse

import numpy as np

import realla_common as rc

try:
    import torch
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:  # allow --help / tests without torch
    DEFAULT_DEVICE = "cpu"


class TheReallaLite:
    def __init__(self, weights_dir="weights", device=DEFAULT_DEVICE):
        self.device = device
        self.flashsr = rc.load_flashsr(weights_dir, device)
        print(f"[TheRealla Lite] Ready on {device}")

    def enhance(self, input_path, output_path, target_lufs=None, subtype="PCM_24"):
        audio, sr = rc.load_audio(input_path)
        audio = rc.resample(audio, sr)
        audio = rc.super_resolve(audio, self.flashsr, self.device)
        if target_lufs is not None:
            audio = rc.loudness_normalize(audio, target_lufs)
        rc.save_audio(output_path, audio, subtype)
        print(f"[TheRealla Lite] Saved: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", default="weights")
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--lufs", type=float, default=None, help="Target LUFS (e.g. -16)")
    p.add_argument("--subtype", default="PCM_24", help="WAV subtype: PCM_16, PCM_24, FLOAT")
    args = p.parse_args()
    TheReallaLite(weights_dir=args.weights, device=args.device).enhance(
        args.input, args.output, target_lufs=args.lufs, subtype=args.subtype)
