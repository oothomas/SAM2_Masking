"""
Core helpers for the SAM‑2/SAMURAI masking workflow.
"""

from __future__ import annotations
import os
import os.path as osp
import cv2
import numpy as np
from typing import Dict, List, Tuple
from contextlib import nullcontext

import torch
from tqdm import tqdm
from pymediainfo import MediaInfo
import piexif
from PIL import Image

# -----------------------------------------------------------------------------
# Video helpers
# -----------------------------------------------------------------------------
def convert_mov_to_mp4(mov_path: str, mp4_path: str | None = None) -> str:
    """Convert .mov -> .mp4 using OpenCV (video only, no audio)."""
    if mp4_path is None:
        stem = osp.splitext(osp.basename(mov_path))[0]
        mp4_path = osp.join(osp.dirname(mov_path), f"{stem}_converted.mp4")

    cap = cv2.VideoCapture(mov_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {mov_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        mp4_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)

    cap.release()
    out.release()
    return mp4_path


def load_video_frames(path: str) -> List[np.ndarray]:
    """Read all frames (BGR) from an .mp4/.mov."""
    if not osp.exists(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def select_roi_on_first_frame(frame: np.ndarray,
                              max_w=800, max_h=600) -> Tuple[int, int, int, int]:
    """Interactively pick an ROI (`cv2.selectROI`)."""
    h, w = frame.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1.0 else frame
    x, y, ww, hh = cv2.selectROI("Select ROI", disp, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    # map back to original resolution
    return int(x / s), int(y / s), int(ww / s), int(hh / s)

# -----------------------------------------------------------------------------
# SAMURAI wrapper
# -----------------------------------------------------------------------------
def _determine_cfg(checkpoint: str) -> str:
    if "large" in checkpoint:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    if "base_plus" in checkpoint:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    if "small" in checkpoint:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    if "tiny" in checkpoint:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    raise ValueError("Cannot infer model size from checkpoint name")


def build_samurai(checkpoint="checkpoints/sam2.1_hiera_large.pt",
                  device="cuda:0"):
    """Returns a ready‑to‑run predictor instance."""
    from sam2.build_sam import build_sam2_video_predictor  # local submodule
    cfg = _determine_cfg(checkpoint)
    return build_sam2_video_predictor(cfg, checkpoint, device=device)


def run_tracking(video_path: str,
                 bbox: Tuple[int, int, int, int],
                 predictor,
                 n_frames: int,
                 device: str = "cuda") -> Dict[int, List[torch.Tensor]]:
    """Zero‑shot tracking. Returns {frame_idx: [mask_tensor, …]}."""
    x, y, w, h = bbox
    autocast_ctx = (torch.autocast("cuda", dtype=torch.float16)
                    if device.startswith("cuda") else nullcontext())
    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(video_path, offload_video_to_cpu=True)
        predictor.add_new_points_or_box(state,
                                        box=(x, y, x + w, y + h),
                                        frame_idx=0, obj_id=0)
        masks = {}
        for fid, _, mask_list in tqdm(
                predictor.propagate_in_video(state),
                total=n_frames - 1,
                desc="SAMURAI propagation"):
            masks[fid] = mask_list
        del state
        torch.cuda.empty_cache()
    return masks

# -----------------------------------------------------------------------------
# EXIF & export helpers
# -----------------------------------------------------------------------------
def _embed_exif(jpg_path: str,
                focal_mm: float | None,
                focal35: int | None,
                make="Apple",
                model="iPhone"):
    """Inject minimal EXIF so that photogrammetry tools stop complaining."""
    img = Image.open(jpg_path)
    exif = {"0th": {}, "Exif": {}, "1st": {}, "thumbnail": None, "GPS": {}}
    if make:
        exif["0th"][piexif.ImageIFD.Make] = make
    if model:
        exif["0th"][piexif.ImageIFD.Model] = model
    if focal_mm:
        exif["Exif"][piexif.ExifIFD.FocalLength] = (int(focal_mm * 100), 100)
    if focal35:
        exif["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(focal35)
    img.save(jpg_path, exif=piexif.dump(exif))


def _metadata_from_video(path: str):
    focal_mm = focal35 = None
    make = "Apple"; model = "iPhone"
    mi = MediaInfo.parse(path)
    for t in mi.tracks:
        if t.track_type != "Video":
            continue
        if t.focal_length:
            try: focal_mm = float(t.focal_length)
            except Exception: pass
        if t.focal_length_in_35mm_format:
            try: focal35 = int(t.focal_length_in_35mm_format)
            except Exception: pass
        if t.make:  make  = t.make
        if t.model: model = t.model
        break
    return focal_mm, focal35, make, model


def save_processed(video_path: str,
                   orig_frames: List[np.ndarray],
                   masks: Dict[int, List[torch.Tensor]]):
    """Dump original / masked frames + binary masks to disk."""
    stem = osp.splitext(osp.basename(video_path))[0]
    out_dir      = osp.join(osp.dirname(video_path), stem)
    orig_outdir  = osp.join(out_dir, "original")
    masked_outdir = osp.join(out_dir, "masked")
    os.makedirs(orig_outdir,  exist_ok=True)
    os.makedirs(masked_outdir, exist_ok=True)

    focal_mm, focal35, make, model = _metadata_from_video(video_path)

    for i, frame in enumerate(orig_frames):
        name = f"{stem}_{i:06d}.jpg"
        orig_path = osp.join(orig_outdir, name)
        cv2.imwrite(orig_path, frame)
        _embed_exif(orig_path, focal_mm, focal35, make, model)

        # build binary mask (H,W) uint8 0/255
        m = np.zeros(frame.shape[:2], np.uint8)
        if i in masks and masks[i]:
            m = (masks[i][0].cpu().numpy()[0] > 0.5).astype(np.uint8) * 255

        mask_path = osp.join(masked_outdir, f"{stem}_{i:06d}_mask.jpg")
        cv2.imwrite(mask_path, m)
        _embed_exif(mask_path, focal_mm, focal35, make, model)

        # masked RGB
        masked_rgb = frame.copy()
        masked_rgb[m == 0] = 0
        rgb_path = osp.join(masked_outdir, name)
        cv2.imwrite(rgb_path, masked_rgb)
        _embed_exif(rgb_path, focal_mm, focal35, make, model)

    print(f"✔ Processed frames saved to: {out_dir}")

# -----------------------------------------------------------------------------
# Key‑frame selection utilities (same as prototype)
# -----------------------------------------------------------------------------
from concurrent.futures import ProcessPoolExecutor

def _lap_var(idx_frame):
    idx, f = idx_frame
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return idx, cv2.Laplacian(g, cv2.CV_64F).var()


def laplacian_stats(frames: List[np.ndarray],
                    subsample=1,
                    workers=8):
    idxs = list(range(0, len(frames), subsample))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(tqdm(ex.map(_lap_var,
                               [(i, frames[i]) for i in idxs]),
                        total=len(idxs),
                        desc="Laplacian var"))
    out.sort(key=lambda t: t[0])
    return zip(*out)   # -> idx_list, var_list


def select_frames(idxs, vars_, thresh=100., min_gap=5):
    chosen = []; last = -1_000
    for i, v in zip(idxs, vars_):
        if v < thresh:           continue
        if i - last < min_gap:   continue
        chosen.append(i); last = i
    return chosen


def detect_keyframes(frames, masks, ratio=0.8):
    """ORB/BFMatcher‑based key‑frame detection (prototype logic)."""
    if not frames: return [], []
    def _mask(idx):
        if idx in masks and masks[idx]:
            m = (masks[idx][0].cpu().numpy()[0] > 0.5).astype(np.uint8)
        else:
            m = np.zeros(frames[0].shape[:2], np.uint8)
        return m
    orb = cv2.ORB_create(); bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    kf, kidx = [frames[0]], [0]
    ref_kp, ref_des = orb.detectAndCompute(frames[0], _mask(0))
    for idx, fr in enumerate(frames[1:], 1):
        kp, des = orb.detectAndCompute(fr, _mask(idx))
        if not (ref_des is not None and des is not None and ref_kp):
            kf.append(fr); kidx.append(idx); ref_kp, ref_des = kp, des; continue
        matches = bf.match(ref_des, des)
        if len(matches) / len(ref_kp) < ratio:
            kf.append(fr); kidx.append(idx); ref_kp, ref_des = kp, des
    return kf, kidx
