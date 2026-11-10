"""TheRealla Pro ("Dimension Man"): denoise -> FlashSR -> EQ -> de-ess -> (gain trim) ->
widen -> loudness.

Channel handling: mono in -> one pass, optional pseudo-stereo widening.
Stereo in -> every stage runs on L and R independently; widening is skipped unless
--widen-stereo-input is given.
Stage order: resample to 48 kHz, denoise, then FlashSR. Denoising happens before SR (DeepFilterNet needs
48 kHz, so it sees an upsampled signal whose top band is empty from a 16 kHz source). Whether SR-then-denoise
sounds better is untested. --device does not reach DeepFilterNet; init_df() chooses its own device.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import iirfilter, sosfilt

import realla_common as rc

try:
    import torch
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None
    DEFAULT_DEVICE = "cpu"

try:  # API per DeepFilterNet docs; verify: python -c "import df.enhance as e; help(e.enhance)"
    from df.enhance import init_df, enhance as dfn_enhance
    HAS_DFN = True
except ImportError:
    HAS_DFN = False


HAAS_MS = 10   # delay for pseudo-stereo; note: summing to mono comb-filters


def _fit(y, n):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


class DimensionManPro:
    def __init__(self, weights_dir="weights", device=DEFAULT_DEVICE, denoise="deepfilter",
                 use_flashsr=True, use_loudnorm=True, target_lufs=-16.0, use_eq=True,
                 use_compression=False, use_desser=True, use_stereo_widen=True,
                 widen_stereo_input=False, dog_bitch=False, subtype="PCM_24"):
        self.device = device
        self.denoise_mode = denoise
        self.use_flashsr = use_flashsr
        self.use_loudnorm = use_loudnorm
        self.target_lufs = target_lufs
        self.use_eq = use_eq
        self.use_compression = use_compression
        self.use_desser = use_desser
        self.use_stereo_widen = use_stereo_widen
        self.widen_stereo_input = widen_stereo_input
        self.dog_bitch = dog_bitch
        self.subtype = subtype
        self.desser_amount = 0.75 if dog_bitch else 0.5

        if self.denoise_mode == "deepfilter":
            if HAS_DFN:
                print("[Pro] Loading DeepFilterNet...")
                self.dfn_model, self.dfn_state, _ = init_df()
            else:
                print("[Pro] WARNING: deepfilternet not installed; denoising disabled "
                      "(pip install deepfilternet, or use --denoise noisereduce)")
                self.denoise_mode = "off"
        elif self.denoise_mode == "noisereduce":
            try:
                import noisereduce as nr
                self.nr = nr
            except ImportError:
                print("[Pro] WARNING: noisereduce not installed; denoising disabled")
                self.denoise_mode = "off"

        self.flashsr = rc.load_flashsr(weights_dir, device) if use_flashsr else None
        self.sos_low = iirfilter(2, 50, btype="low", ftype="bessel", output="sos", fs=rc.SR)
        self.sos_high = iirfilter(2, 8000, btype="high", ftype="bessel", output="sos", fs=rc.SR)
        self.sos_desser = iirfilter(2, 7000, btype="high", ftype="bessel", output="sos", fs=rc.SR)

    # ---------- DSP stages (all operate on 1-D float arrays) ----------
    def _apply_eq(self, x):
        """Adds 30 % of the <50 Hz band and 20 % of the >8 kHz band. Measured: about +1.8 dB at 20 Hz,
        +0.8 dB at 12 kHz, flat at 1 kHz (2nd-order Bessel slopes are gentle)."""
        return x + 0.3 * sosfilt(self.sos_low, x, axis=0) + 0.2 * sosfilt(self.sos_high, x, axis=0)

    def _apply_desser(self, x):
        """Static high-band cut (not a dynamic de-esser): subtracts a share of the >7 kHz band.
        Measured at amount 0.5: about -2 dB at 12 kHz, flat at 1 kHz."""
        return x - self.desser_amount * sosfilt(self.sos_desser, x, axis=0)

    def _per_channel(self, fn, audio):
        return np.column_stack([fn(audio[:, c]) for c in range(audio.shape[1])]).astype(np.float32)

    def _denoise(self, audio):
        if self.denoise_mode == "off":
            return audio
        cols = []
        for c in range(audio.shape[1]):
            x = audio[:, c]
            if self.denoise_mode == "deepfilter":
                t = torch.from_numpy(x).unsqueeze(0)
                y = dfn_enhance(self.dfn_model, self.dfn_state, t).squeeze(0).cpu().numpy()
            else:  # noisereduce: assumes the first 0.5 s is noise-only
                y = self.nr.reduce_noise(y=x, sr=rc.SR, y_noise=x[:int(0.5 * rc.SR)],
                                         prop_decrease=0.95 if self.dog_bitch else 0.85)
            cols.append(_fit(y, len(x)))
        return np.column_stack(cols)

    def _apply_global_gain_trim(self, audio, threshold=-12.0, ratio=4.0):
        """TODO: not a real compressor. One static gain from whole-file RMS (no envelope,
        attack/release or makeup). Off by default."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        db = 20 * np.log10(rms + 1e-12)
        gain_db = (threshold - db) * (1 - 1 / ratio) if db > threshold else 0.0
        return audio * (10 ** (gain_db / 20))

    def _stereo_widen(self, audio):
        if audio.shape[1] == 1:
            if not self.use_stereo_widen:
                return audio
            x = audio[:, 0]
            d = int(HAAS_MS * rc.SR / 1000)
            delayed = np.concatenate([np.zeros(d, np.float32), x[:-d]]) if len(x) > d else x
            return np.column_stack((x, delayed))   # equal gain, so no level imbalance
        if self.widen_stereo_input:
            mid = (audio[:, 0] + audio[:, 1]) / 2
            side = (audio[:, 0] - audio[:, 1]) / 2 * 1.5
            return np.column_stack((mid + side, mid - side))
        return audio

    # ---------- pipeline ----------
    def process(self, input_path, output_path):
        print(f"\n[Dimension Man Pro] Processing: {input_path}")
        audio, sr = rc.load_audio(input_path)
        print(f"  [Pro] {audio.shape[1]} channel(s), {sr} Hz")
        audio = rc.resample(audio, sr)
        audio = self._denoise(audio)
        if self.flashsr is not None:
            print("  [Pro] FlashSR upscaling (chunked, per channel)...")
            audio = rc.super_resolve(audio, self.flashsr, self.device)
        if self.use_eq:
            audio = self._per_channel(self._apply_eq, audio)
        if self.use_desser:
            audio = self._per_channel(self._apply_desser, audio)
        if self.use_compression:
            audio = self._apply_global_gain_trim(audio)
        audio = self._stereo_widen(audio)
        if self.use_loudnorm:
            print(f"  [Pro] Normalising to {self.target_lufs} LUFS...")
            audio = rc.loudness_normalize(audio, self.target_lufs)
        rc.save_audio(output_path, audio, self.subtype)
        print(f"[TheRealla Pro] Exported: {output_path}")

    def process_batch(self, input_dir, output_dir, exts=(".wav", ".flac")):
        from tqdm import tqdm
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"--batch needs --input to be a directory: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in exts)
        seen = set()
        for f in tqdm(files, desc="Batch processing"):
            name = f.stem + ".wav"
            if name in seen:   # e.g. a.wav and a.flac in the same folder
                name = f"{f.stem}_{f.suffix[1:].lower()}.wav"
            seen.add(name)
            self.process(f, output_dir / name)


def main():
    p = argparse.ArgumentParser(description="TheRealla Pro")
    p.add_argument("--input", required=True, help="File or directory")
    p.add_argument("--output", required=True, help="Output file or directory")
    p.add_argument("--weights", default="weights")
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--denoise", choices=["deepfilter", "noisereduce", "off"], default="deepfilter")
    p.add_argument("--no-flashsr", action="store_false", dest="flashsr")
    p.add_argument("--no-loudnorm", action="store_false", dest="loudnorm")
    p.add_argument("--target-lufs", type=float, default=-16.0)
    p.add_argument("--no-eq", action="store_false", dest="eq")
    p.add_argument("--compression", action="store_true", help="Enable static gain trim (not a true compressor)")
    p.add_argument("--no-desser", action="store_false", dest="desser")
    p.add_argument("--no-stereo", action="store_false", dest="stereo", help="Keep mono input mono")
    p.add_argument("--widen-stereo-input", action="store_true", help="Also widen already-stereo input")
    p.add_argument("--dog-bitch", action="store_true", help="Max aggression: stronger denoise and de-ess")
    p.add_argument("--subtype", default="PCM_24", help="WAV subtype: PCM_16, PCM_24, FLOAT")
    p.add_argument("--batch", action="store_true")
    a = p.parse_args()

    plugin = DimensionManPro(
        weights_dir=a.weights, device=a.device, denoise=a.denoise, use_flashsr=a.flashsr,
        use_loudnorm=a.loudnorm, target_lufs=a.target_lufs, use_eq=a.eq,
        use_compression=a.compression, use_desser=a.desser, use_stereo_widen=a.stereo,
        widen_stereo_input=a.widen_stereo_input, dog_bitch=a.dog_bitch, subtype=a.subtype)
    if a.batch:
        plugin.process_batch(a.input, a.output)
    else:
        plugin.process(a.input, a.output)


if __name__ == "__main__":
    main()
