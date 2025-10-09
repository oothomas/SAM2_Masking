# 🎬 SAM‑2 / SAMURAI Video‑Masking Pipeline
_A zero‑shot CLI that converts raw video into photogrammetry‑ready image sets_

---

## 1 Why does this exist?
Building a 3‑D model from **video** usually starts with a slog:

1. Extract every frame  
2. Mask each frame by hand  
3. Rename files and inject camera EXIF so WebODM / RealityCapture will accept them  

This repo turns that entire workflow into **one command** by wrapping
Meta AI’s state‑of‑the‑art video segmenter **SAMURAI (SAM‑2)** with a lean Python CLI.

> **Result**  
> Draw _one_ box → receive three tidy folders:
>
> ```
> original/   # untouched RGB JPEGs
> masked/     # background‑removed RGB + *_mask.jpg (binary)
> keyframes/  # optional, if you enable key‑frame extraction
> ```

---

## 2 What are SAM‑2 and SAMURAI?
| Component | Role | Links |
|-----------|------|-------|
| **SAM‑2** | Extends Segment‑Anything from images to spatio‑temporal video masks | <https://github.com/facebookresearch/sam2> |
| **SAMURAI** | Official implementation + pretrained weights built on SAM‑2 | <https://github.com/yangchris11/samurai> |

The script `sam2/setup_samurai.sh` clones the full SAMURAI repository into
`sam2/`. This repo already vendors the `sam2/` Python package that your imports
rely on—so after running the setup script you don’t need to juggle two separate
check‑outs.

---

## 3 Key features
* **Codec‑agnostic input** – auto‑converts `.mov` → `.mp4` if needed  
* **Single‑click ROI** – draw a bounding box on the first frame, then relax  
* **Zero‑shot multi‑object tracking** – SAMURAI propagates masks frame‑by‑frame  
* **Photogrammetry‑friendly export**
  * `original/` – raw JPEGs with focal length, 35 mm eq., make/model in EXIF  
  * `masked/`  – background‑removed JPEG + binary mask  
* **Optional key‑frame pickers** – Laplacian variance (blur filter) or ORB matcher
* **Pure CLI** – no Jupyter dependencies  
* **CUDA‑accelerated** – float‑16 inference for fast GPUs (falls back to CPU)

---

## 4 Quick start

```bash
# 1 Clone this repo
git clone https://github.com/<you>/sam2-masking-pipeline.git
cd sam2-masking-pipeline

# 2 Bootstrap SAMURAI source + checkpoints (default = large model)
#    (clones the SAMURAI repo into sam2/ if needed)
bash sam2/setup_samurai.sh               # or bash sam2/setup_samurai.sh small

# 3 Create an isolated Python env (3.9 – 3.11)
conda create -n sam2 python=3.10 -y
conda activate sam2

# 4 Install PyTorch that matches your GPU – example: CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5 Other Python deps
pip install -r requirements.txt

# 6 Run 🎉
python -m sam2_masking \
       --video /path/to/MyScan.mov \
       --checkpoint checkpoints/sam2.1_hiera_large.pt \
       --device cuda:0
```

A window appears—draw a tight box around the object.  
When the progress bar finishes you will find:

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
| **GPU drivers** | NVIDIA 535 + for CUDA 12.1 wheels | same |
| **Python** | 3.9 – 3.11 via conda/pyenv | Miniconda recommended |

> **CPU‑only?** Install the CPU wheels:  
> `pip install torch==2.5.* torchvision==0.20.*` and run with `--device cpu`.

---

## 6 CLI reference

```
python -m sam2_masking --video <file> [options]

Required
  --video PATH            .mov / .mp4 file

Common options
  --checkpoint PATH       Path to a SAMURAI weight file
  --device cuda:0|cpu     Compute device
  --no-convert            Skip the automatic MOV→MP4 step
  --keyframe-threshold F  Enable ORB/BFMatcher key-frame thinning
  --quality-threshold F   Drop blurry frames using Laplacian variance >= F
  --quality-min-gap N     Require at least N frames between quality-filter hits

For the full list:
  python -m sam2_masking --help
```

---

## 7 Project structure

```
sam2_masking/           # installable Python package
  ├─ core.py            # processing logic
  ├─ cli.py             # CLI helpers
  └─ __main__.py        # entry point for `python -m sam2_masking`

sam2/                   # SAMURAI repository cloned by setup_samurai.sh
                        # (includes the sam2 package)
checkpoints/            # large .pt weight files – ignored by git
docs/                   # demo GIFs / screenshots (optional)
```

---

## 8 Extending the pipeline
* **Multiple objects** – call `predictor.add_new_points_or_box()` per instance.  
* **Key‑frames** – import `laplacian_stats` or `detect_keyframes`
  from `sam2_masking.core` and bolt them into your script.  
* **Fine‑tuning** – drop in your own checkpoint; config is inferred from
  the filename suffix (`*_large.pt`, `*_base_plus.pt`, etc.).

---

## 9 FAQ

| Question | Answer |
|----------|--------|
| Does audio survive the MOV→MP4 step? | No—OpenCV writes video‑only. |
| How big is the “large” checkpoint? | ≈ 2 GB (base ≈ 600 MB, tiny ≈ 250 MB). |
| RTX 20‑series support? | Yes; FP16 needs compute 7.0+. Otherwise use `--device cpu`. |
| Official Meta support? | No—community wrapper. SAM‑2 & SAMURAI are © Meta AI. |

---

## 10 Cite SAM2 and SAMURAI

If this tool aids your research, please cite SAM‑2 / SAMURAI:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@misc{yang2024samurai,
  title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory}, 
  author={Cheng-Yen Yang and Hsiang-Wei Huang and Wenhao Chai and Zhongyu Jiang and Jenq-Neng Hwang},
  year={2024},
  eprint={2411.11922},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.11922}, 
}
```

---

## 11 License
* **Wrapper code:** MIT  
* **SAMURAI (sam2/) source:** Apache 2.0  
* **Model checkpoints:** see SAMURAI licence

Enjoy painless video masking! 🚀
