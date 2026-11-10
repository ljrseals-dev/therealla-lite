# Weights

No weight files are stored in this repository. The MIT license in `LICENSE` covers the wrapper code only. Third-party weights keep their own terms, which cannot be changed by adding them to an MIT repository.

## FlashSR (required by `Lite.py` and `Pro.py`)

| File | Placed in |
|---|---|
| `student_ldm.pth` | `weights/` |
| `sr_vocoder.pth` | `weights/` |
| `vae.pth` | `weights/` |

- **Source:** the reference repo's Readme links https://huggingface.co/datasets/jakeoneijk/FlashSR_weights (a Hugging Face *dataset* repository). Check each file's size against that listing after download; a truncated or Git-LFS-pointer file fails confusingly.
- **License:** the reference repository contains no top-level license file and its Readme states none. A third-party port says the weights are CC-BY-NC 4.0 (non-commercial). Treat the weights as non-commercial until the authors (Jaekwon Im, Juhan Nam) say otherwise.
- **Loading note:** the reference code calls `torch.load` without `map_location`, so loading on a CPU-only machine may fail if the checkpoints were saved from CUDA. Untested.

## musicnn tagger (separate from TheRealla)

`MSD_musicnn_optimized_fp16.pt` is a float16 export of the MSD-trained musicnn model, in the naming used by a community PyTorch port. The architecture and original weights come from Pons and Serra. Their license terms were not checked here and apply to this file; do not label it MIT unless the upstream terms allow it.

## If you want to release your own weights under MIT

Only weights you trained yourself, or whose upstream license permits relicensing, can be marked MIT. Put a `LICENSE-weights` file next to them, name the training data, and state any data-license restrictions.
