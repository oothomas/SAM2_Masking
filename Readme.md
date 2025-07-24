# 🎬 SAM‑2 / SAMURAI Video‑Masking Pipeline
_A zero‑shot CLI for turning raw video into photogrammetry‑ready image sets_

---

## 1 Why does this exist?
Building a 3‑D model from **video** usually starts with a slog:

1. Extract every frame.
2. Mask your object by hand.
3. Rename files and inject camera EXIF so WebODM/RealityCapture will accept them.

This repo turns that entire workflow into **one command** by wrapping
Meta AI’s state‑of‑the‑art video segmenter
**SAMURAI (SAM‑2)** with a lightweight Python CLI.

> **Result**  
> Draw _one_ box → receive three tidy folders:
>
> ```
> original/   # untouched RGB JPEGs
> masked/     # foreground‑only RGB + *_mask.jpg (binary)
> keyframes/  # optional (if enabled)
> ```

---

## 2 What are SAM‑2 and SAMURAI?
| Component | Role | Links |
|-----------|------|-------|
| **SAM‑2** | Extends Segment‑Anything from images to spatio‑temporal video masks | <https://github.com/facebookresearch/sam2> |
| **SAMURAI** | Pretrained weights + hierarchical decoder built on SAM‑2 | <https://yangchris11.github.io/samurai/> |

Our pipeline downloads SAMURAI checkpoints and drives them
directly—no need to juggle multiple demos.

---

## 3 Key features
* **Codec‑agnostic input** – auto‑converts `.mov` → `.mp4` if needed  
* **Single‑click ROI** – select a bounding box on the first frame, then relax  
* **Zero‑shot multi‑object tracking** – SAMURAI propagates masks frame‑by‑frame  
* **Photogrammetry‑friendly export**
  * `original/` – raw JPEGs with focal length, 35 mm eq., make/model in EXIF  
  * `masked/`  – background‑removed JPEG + binary mask  
* **Optional key‑frame pickers** – Laplacian variance or ORB matcher  
* **Pure CLI** – no Jupyter dependencies  
* **CUDA‑accelerated** – float‑16 inference for fast GPUs (falls back to CPU)

---

## 4 Quick start

```bash
# Clone and pull the SAMURAI submodule
git clone https://github.com/<you>/sam2-masking-pipeline.git
cd sam2-masking-pipeline
git submodule update --init --recursive   # brings facebookresearch/sam2 into ./sam2

# Create an isolated Python env (3.9 – 3.11)
conda create -n sam2 python=3.10 -y
conda activate sam2

# Install PyTorch that matches your GPU – example: CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Remaining dependencies
pip install -r requirements.txt

# Download a SAMURAI checkpoint (~2 GB) into ./checkpoints/
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt \
     https://dl.fbaipublicfiles.com/samurai/models/sam2.1_hiera_large.pt

# Run 🎉
python -m sam2_masking \
       --video /path/to/MyScan.mov \
       --checkpoint checkpoints/sam2.1_hiera_large.pt \
       --device cuda:0
```

A window will appear—draw a tight box around the object.  
When the progress bar completes you will find:

```
MyScan/
 ├─ original/
 ├─ masked/
 └─ keyframes/   # only if you enable key‑frame extraction
```

---

## 5 Detailed installation notes

| Item | Linux / macOS | Windows |
|------|---------------|---------|
| **System packages** | `sudo apt install ffmpeg mediainfo libgl1` | `choco install ffmpeg mediainfo` |
| **GPU drivers** | NVIDIA 535 + for CUDA 11.8 | Same |
| **Python** | 3.9 – 3.11 via conda/pyenv | Miniconda recommended |

> **CPU‑only?** Use `pip install torch==2.2.*` (no CUDA wheel) and run with `--device cpu`.

---

## 6 CLI reference

```
python -m sam2_masking --video <file> [options]

Required
  --video PATH            .mov / .mp4 input

Common options
  --checkpoint PATH       SAMURAI weight file (default: checkpoints/*large.pt)
  --device cuda:0|cpu     Compute device
  --no-convert            Skip the automatic MOV→MP4 step

For full list:
  python -m sam2_masking --help
```

---

## 7 Project structure

```
sam2_masking/           # installable Python package
  ├─ core.py            # processing logic
  └─ cli.py             # entry‑point (python -m sam2_masking)

sam2/                   # SAM‑2 source (git submodule)
checkpoints/            # large .pt files – ignored by git
docs/                   # demo GIFs / screenshots (optional)
```

---

## 8 Extending the pipeline
* **Multiple objects:** call `predictor.add_new_points_or_box()` per instance.  
* **Key‑frames:** import `laplacian_stats` or `detect_keyframes` from
  `sam2_masking.core` and bolt them into your script.  
* **Fine‑tuning:** drop in your own checkpoint; config is inferred from
  filename suffix (`*_large.pt`, `*_base_plus.pt`, etc.).

---

## 9 FAQ

| Question | Answer |
|----------|--------|
| Does audio survive the MOV→MP4 step? | No—OpenCV writes video‑only. |
| How big is the “large” checkpoint? | ≈ 2 GB (base ≈ 600 MB, tiny ≈ 250 MB). |
| RTX 20‑series support? | Yes; FP16 needs compute 7.0+. Otherwise use `--device cpu`. |
| Official Meta support? | No—community wrapper. SAM‑2 & SAMURAI are © Meta AI. |

---

## 10 Cite us

If this tool aids your research, please cite SAM‑2 / SAMURAI:

```bibtex
@inproceedings{kirillov2025sam2,
  title     = {Segment Anything in Video},
  author    = {Kirillov, A. and He, K. and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

---

## 11 License
* **Wrapper code:** MIT  
* **SAM‑2 submodule:** Apache 2.0  
* **Model checkpoints:** see SAMURAI licence

Enjoy painless video masking! 🚀
