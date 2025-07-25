"""
Command‑line entry point.

Example
-------
$ python -m sam2_masking \
      --video /data/foo.mov \
      --checkpoint checkpoints/sam2.1_hiera_large.pt \
      --device cuda:0
"""

import argparse
import sys

import torch

from .core import (
    convert_mov_to_mp4,
    load_video_frames,
    select_roi_on_first_frame,
    build_samurai,
    run_tracking,
    save_processed,

)


def parse_args():
    p = argparse.ArgumentParser(description="SAM‑2 SAMURAI masking pipeline")
    p.add_argument("--video", required=True, help=".mov or .mp4 input file")
    p.add_argument("--checkpoint",
                   default="checkpoints/sam2.1_hiera_large.pt",
                   help="Path to a SAMURAI checkpoint (.pt)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-convert", action="store_true",
                   help="Skip .mov -> .mp4 conversion even if input is .mov")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. MOV → MP4 (OpenCV writer does video‑only, no audio)
    in_path = args.video
    if in_path.lower().endswith(".mov") and not args.no_convert:
        print("Converting .mov → .mp4 ...")
        in_path = convert_mov_to_mp4(in_path)

    # 2. Frame buffer
    frames = load_video_frames(in_path)
    print(f"Loaded {len(frames)} frames from {in_path}")

    # 3. ROI UI on first frame
    bbox = select_roi_on_first_frame(frames[0])
    print(f"ROI = (x,y,w,h): {bbox}")

    # 4. Predictor
    predictor = build_samurai(args.checkpoint, args.device)

    # 5. Tracking
    masks = run_tracking(in_path, bbox, predictor, len(frames), args.device)

    # 6. Export (original + masked + binary masks)
    save_processed(args.video, frames, masks)

    print("🎉 Done!")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
