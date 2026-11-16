"""Extract a crude, zero-model "interaction" proxy from FindingDory episode
videos: spatial concentration of frame-to-frame pixel change.

This is a deliberately simple approximation of P07's interaction-detection
signal (AMEGO's online hand-object interaction tracklets, learned from visual
features). Here there is no object/hand detection at all -- just a heuristic:
split each frame-to-frame absolute-difference map into a grid of blocks and
score the frame by the STANDARD DEVIATION of per-block mean change (not the
overall mean, which is what src/scripts/extract_pixel_novelty.py already
computes). The intuition: a hand/object manipulation tends to change a small,
localized region against an otherwise-static background (high spatial
variance of change), whereas the robot walking or turning its head tends to
change the whole frame roughly uniformly (low spatial variance of change,
even if the overall magnitude is similar). This is a hand-picked heuristic,
not derived or validated against any ground-truth interaction labels, and is
expected to also fire on things that aren't interactions (e.g. a small moving
object elsewhere in the scene, lighting changes in one area).

Output: same format as extract_pixel_novelty.py -- one CSV with columns
[ep_id, frame_idx, score], one row per frame. `score` is the standard
deviation (0-255 scale) of block-mean absolute pixel difference from the
previous frame, over an 8x8 grid on a 64x64 downsampled grayscale frame;
frame 0 of each episode gets score 0.0 by convention.

Usage:
    python3 extract_motion_concentration.py <video_dir> <output_csv> [--resize 64] [--grid 8]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def score_video(path: Path, resize: int, grid: int) -> list[float]:
    cap = cv2.VideoCapture(str(path))
    scores: list[float] = []
    prev_gray = None
    block = resize // grid
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev_gray is None:
            scores.append(0.0)
        else:
            diff = np.abs(gray - prev_gray)
            # Reshape into a (grid, block, grid, block) view and average within
            # each block to get a `grid x grid` map of local change.
            block_means = diff.reshape(grid, block, grid, block).mean(axis=(1, 3))
            scores.append(float(block_means.std()))
        prev_gray = gray
    cap.release()
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("output_csv", type=Path)
    ap.add_argument("--resize", type=int, default=64, help="must be divisible by --grid")
    ap.add_argument("--grid", type=int, default=8, help="grid x grid blocks")
    args = ap.parse_args()
    if args.resize % args.grid != 0:
        print("--resize must be divisible by --grid", file=sys.stderr)
        return 1

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        print(f"no .mp4 files found in {args.video_dir}", file=sys.stderr)
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with args.output_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ep_id", "frame_idx", "score"])
        for i, vid in enumerate(videos):
            ep_id = vid.stem
            scores = score_video(vid, args.resize, args.grid)
            for idx, s in enumerate(scores):
                w.writerow([ep_id, idx, f"{s:.4f}"])
            if (i + 1) % 10 == 0 or (i + 1) == len(videos):
                print(f"  {i + 1}/{len(videos)} videos, {len(scores)} frames in {ep_id}, "
                      f"{time.time() - t0:.0f}s elapsed", flush=True)

    print(f"wrote {args.output_csv} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
