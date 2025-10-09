"""
Core helpers for the SAM‑2/SAMURAI masking workflow.
"""

from __future__ import annotations
import os
import os.path as osp
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor

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

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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


def select_roi_on_first_frame(
        frame: np.ndarray,
        max_w: int = 800,
        max_h: int = 600,
) -> Tuple[int, int, int, int]:
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

    import os, sys, pathlib

    ROOT = pathlib.Path(__file__).resolve().parent.parent  # repo root
    SAMURAI_PKG = ROOT / "samurai_src" / "sam2"  # adjust if you renamed

    if SAMURAI_PKG.exists() and str(SAMURAI_PKG) not in sys.path:
        sys.path.insert(0, str(SAMURAI_PKG))

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

embed_exif_into_jpg = _embed_exif

def _metadata_from_video(path: str):
    focal_mm = focal35 = None
    make = "Apple"
    model = "iPhone"
    mi = MediaInfo.parse(path)
    for t in mi.tracks:
        if t.track_type != "Video":
            continue
        if t.focal_length:
            try:
                focal_mm = float(t.focal_length)
            except Exception:
                pass
        if t.focal_length_in_35mm_format:
            try:
                focal35 = int(t.focal_length_in_35mm_format)
            except Exception:
                pass
        if t.make:
            make = t.make
        if t.model:
            model = t.model
        break
    return focal_mm, focal35, make, model


def save_processed(video_path,
                   frames,
                   all_masks,
                   frame_indices: Optional[List[int]] = None):
    """
    Given:
      - video_path: path to the input .mp4 (or .mov) file.
      - frames: a list of frames (images in BGR format). This list can be a
                subset of the original timeline when key-frame thinning is
                enabled.
      - all_masks: a dict mapping frame indices to lists of mask tensors.
                   We assume each list contains at least one mask; if none, the frame is unmasked.
      - frame_indices: optional list of indices representing the position of
                       each frame within the original timeline. When omitted
                       the frames are assumed to be sequential starting at 0.

    This function creates an output folder (named after the video file's stem, in the same directory as the video)
    with two subfolders:
      - "original": containing the original frames saved as JPEGs.
      - "masked": containing both the frames with the background removed (only pixels where the mask==1 are kept)
                  and the binary masks saved as JPEGs with '_mask' appended to the filename.

    It also attempts to extract EXIF-like metadata from the video via pymediainfo and embed
    that data into each saved JPEG.
    """

    # 1) Extract metadata from video
    focal_length = None
    focal_length_35mm = None
    make = "Apple"
    model = "iPhone"

    media_info = MediaInfo.parse(video_path)
    for track in media_info.tracks:
        if track.track_type == "Video":
            if track.focal_length:
                try:
                    focal_length = float(track.focal_length)
                except Exception:
                    pass
            if track.focal_length_in_35mm_format:
                try:
                    focal_length_35mm = int(track.focal_length_in_35mm_format)
                except Exception:
                    pass
            if track.make:
                make = track.make
            if track.model:
                model = track.model
            break

    # 2) Determine the video directory and filestem
    video_dir = os.path.dirname(video_path)
    video_filename = os.path.basename(video_path)
    video_stem, _ = os.path.splitext(video_filename)

    # 3) Create the output folder (named after the video file's stem) and subfolders:
    #    "original" for original frames, and "masked" for both masked frames and masks.
    out_dir = os.path.join(video_dir, video_stem)
    original_dir = os.path.join(out_dir, "original")
    masked_dir = os.path.join(out_dir, "masked")

    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)

    # 4) Iterate through frames and save them
    for idx, frame in enumerate(frames):
        frame_idx = frame_indices[idx] if frame_indices is not None else idx
        # Build the base name: e.g. video_stem_000001
        out_basename = f"{video_stem}_{frame_idx:06d}"

        # ---------------------
        # (A) Save the original frame
        # ---------------------
        orig_filename = os.path.join(original_dir, f"{out_basename}.jpg")
        cv2.imwrite(orig_filename, frame)

        # Inject EXIF metadata into the original image
        embed_exif_into_jpg(
            orig_filename,
            focal_mm=focal_length,
            focal35=focal_length_35mm,
            make=make,
            model=model
        )

        # ---------------------
        # (B) Get or build the mask
        # ---------------------
        if frame_idx in all_masks and len(all_masks[frame_idx]) > 0:
            mask_tensor = all_masks[frame_idx][0]
            # Convert from PyTorch tensor ([1, H, W]) to a numpy array (H, W)
            mask_np = mask_tensor.cpu().numpy()[0]
            # Threshold the mask to create a binary mask (values 0 or 1)
            binary_mask = (mask_np > 0.5).astype(np.uint8)
        else:
            # If no mask exists, create a binary mask filled with zeros
            binary_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # ---------------------
        # (C) Save the binary mask and the masked frame in the same folder ("masked")
        # ---------------------

        # Save the binary mask image (scale 0/1 to 0/255)
        mask_filename = os.path.join(masked_dir, f"{out_basename}_mask.jpg")
        mask_to_save = (binary_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_filename, mask_to_save)
        embed_exif_into_jpg(
            mask_filename,
            focal_mm=focal_length,
            focal35=focal_length_35mm,
            make=make,
            model=model
        )

        # Create the masked frame: keep only pixels where binary_mask==1, set background to black.
        binary_mask_3ch = np.stack([binary_mask] * 3, axis=-1)
        masked_frame = frame * binary_mask_3ch

        # Save the masked frame
        masked_filename = os.path.join(masked_dir, f"{out_basename}.jpg")
        cv2.imwrite(masked_filename, masked_frame)
        embed_exif_into_jpg(
            masked_filename,
            focal_mm=focal_length,
            focal35=focal_length_35mm,
            make=make,
            model=model
        )

    print(f"Processed frames saved in folder: {out_dir}")

    # -- OPTIONAL DEBUG PRINT --
    print("Extracted metadata from video:")
    print(f"  Focal length: {focal_length}")
    print(f"  Focal length (35mm): {focal_length_35mm}")
    print(f"  Make: {make}, Model: {model}")

    return media_info


# -----------------------------------------------------------------------------
# Key‑frame selection utilities (same as prototype)
# -----------------------------------------------------------------------------

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
    return zip(*out)  # -> idx_list, var_list


def select_frames(idxs, vars_, thresh=100., min_gap=5):
    chosen = []
    last = -1_000
    for i, v in zip(idxs, vars_):
        if v < thresh:
            continue
        if i - last < min_gap:
            continue
        chosen.append(i)
        last = i
    return chosen


def detect_keyframes(frames, masks, ratio=0.8):
    """ORB/BFMatcher‑based key‑frame detection (prototype logic)."""
    if not frames:
        return [], []

    def _mask(idx):
        if idx in masks and masks[idx]:
            return (masks[idx][0].cpu().numpy()[0] > 0.5).astype(np.uint8)
        return np.zeros(frames[0].shape[:2], np.uint8)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    kf, kidx = [frames[0]], [0]
    ref_kp, ref_des = orb.detectAndCompute(frames[0], _mask(0))
    iterable = enumerate(frames[1:], 1)
    if len(frames) > 1:
        iterable = tqdm(iterable,
                        total=len(frames) - 1,
                        desc="Key-frame thinning",
                        leave=False)
    for idx, fr in iterable:
        kp, des = orb.detectAndCompute(fr, _mask(idx))
        if not (ref_des is not None and des is not None and ref_kp):
            kf.append(fr)
            kidx.append(idx)
            ref_kp, ref_des = kp, des
            continue
        matches = bf.match(ref_des, des)
        if len(matches) / len(ref_kp) < ratio:
            kf.append(fr)
            kidx.append(idx)
            ref_kp, ref_des = kp, des
    return kf, kidx
