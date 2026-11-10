# TheRealla: expensive sound, minimal code

TheRealla turns cheap-microphone recordings into 48 kHz audio using **FlashSR** super-resolution, with an optional processing chain.

## Scripts

| Script | What it does |
|---|---|
| `Lite.py` | Resample, then FlashSR, then optional loudness normalisation |
| `Pro.py` | Resample, denoise, FlashSR, EQ, de-ess, optional gain trim, stereo widening, loudness |
| `realla_common.py` | Shared helpers (both scripts need it in the same folder) |

Both scripts handle audio of any length. FlashSR itself accepts a fixed 5.12 s (245,760 samples at 48 kHz), so audio is processed in 50%-overlapping Hann-windowed chunks and cross-faded. Pad samples are cropped from the output, but the model still sees them, so its output on the first and last chunks can be influenced by the padding (reflect padding for longer inputs, zeros for inputs shorter than one hop). Overlap-add can leave seams or phase drift with a diffusion model, so listen for artifacts on long files.

**Channels:** mono in gives mono out (Pro can add pseudo-stereo widening). Stereo in gives stereo out; each channel goes through the whole chain independently, and widening is skipped unless `--widen-stereo-input` is set. More than two channels are downmixed to mono (6 channels are assumed to be 5.1 and the LFE channel is excluded).

**Output:** 24-bit WAV by default (`--subtype PCM_16 | PCM_24 | FLOAT`). If the final peak would exceed about -0.1 dBFS it is scaled down and a warning is printed; in that case the LUFS target is missed by that amount. Source bit depth and sample rate do not carry extra information past the 48 kHz resampling step.

**Stage order and devices (Pro):** denoising runs at 48 kHz before FlashSR, so DeepFilterNet sees an upsampled signal; SR-then-denoise is untested. `--device` applies to FlashSR only; DeepFilterNet picks its own device. Pseudo-stereo for mono input is an equal-gain 10 ms delayed copy and will comb-filter if summed back to mono.

## Installation

1. Install FlashSR from the reference implementation (https://github.com/jakeoneijk/FlashSR_Inference; its `Script/2_setup.sh` runs `pip install -e ./`) and download its three weight files (`student_ldm.pth`, `sr_vocoder.pth`, `vae.pth`) from the weights page linked in that repo's Readme (https://huggingface.co/datasets/jakeoneijk/FlashSR_weights) into `weights/`. See `WEIGHTS.md` for sources and licenses.
2. Install dependencies:

```bash
# Lite
pip install torch soundfile numpy scipy librosa pyloudnorm
# Pro (adds)
pip install tqdm noisereduce
# Optional premium denoiser (pip name deepfilternet, import name df)
pip install deepfilternet
```

## Usage

```bash
python Lite.py --input mic.wav --output enhanced.wav --lufs -16
python Pro.py --input mic.wav --output pro.wav --denoise noisereduce
python Pro.py --input ./recordings/ --output ./enhanced/ --batch
```

### Lite options
`--input`, `--output`, `--weights` (default `weights`), `--device`, `--lufs`, `--subtype`.

### Pro options

| Argument | Default | Description |
|---|---|---|
| `--denoise` | `deepfilter` | `deepfilter`, `noisereduce`, or `off`. If the chosen engine is not installed, a warning is printed and denoising is disabled |
| `--no-flashsr` | off | Skip super-resolution |
| `--no-loudnorm` / `--target-lufs` | on / -16 | Loudness normalisation |
| `--no-eq` | off | Disable EQ (small boosts below 50 Hz and above 8 kHz) |
| `--no-desser` | off | Disable the static high-band cut above 7 kHz |
| `--compression` | off | Static whole-file gain trim; **not** a true compressor |
| `--no-stereo` | off | Keep mono input mono |
| `--widen-stereo-input` | off | Also widen already-stereo input |
| `--dog-bitch` | off | Stronger noisereduce setting and a stronger high-band cut |
| `--subtype` | `PCM_24` | Output WAV subtype |
| `--batch` | off | `--input` is a directory of `.wav` / `.flac` files |

## Status

Verified without GPU/FlashSR: the chunked overlap-add reproduces its input to about 1e-7 for lengths from 1 sample to 15 s, and the EQ and de-ess stages behave as documented. **Not tested:** FlashSR inference, DeepFilterNet, and the audio I/O libraries; the FlashSR constructor arguments and the `df.enhance` API follow their documentation and should be checked on first run.

## License

The wrapper code in this repository is MIT-licensed (see `LICENSE`). The model weights are not part of this repository and are not covered by that license; see `WEIGHTS.md`.

The wrapper is provided as-is. FlashSR is by Jaekwon Im and Juhan Nam (KAIST). A third-party port states that the FlashSR weights are licensed CC-BY-NC 4.0 (non-commercial); confirm the terms at the upstream repository before any commercial use.
