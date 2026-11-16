"""Extract a crude, zero-model "surprise" proxy from FindingDory episode videos:
per-frame mean absolute pixel difference from the previous frame.

This is a deliberately simple approximation of P01's Bayesian-surprise signal
(online prediction error in a learned latent space). Here the "predictor" is
just the previous frame and the "latent space" is raw downsampled pixels --
there is no learned model at all. It is meant as a first, cheap test of
whether a surprise-like signal helps before investing in anything heavier
(e.g. a real embedding model), per the project's "simple before complex"
discipline (see docs/research_roadmap.md). Label it as `pixel_novelty`, not
as a reproduction of P01's method, everywhere it is used.

Self-contained on purpose (only cv2 + numpy + stdlib) so it can be copied to
and run on a machine that doesn't have this repo's package layout (e.g. via
plain `scp`), such as the remote host holding the actual video files.

Output: one CSV with columns [ep_id, frame_idx, score], one row per frame,
for every <video_dir>/*.mp4. `score` is the mean absolute pixel difference
(0-255 scale, averaged over a downsampled grayscale frame) from the previous
frame; frame 0 of each episode gets score 0.0 by convention (no prior frame
to differ from).

Usage:
    python3 extract_pixel_novelty.py <video_dir> <output_csv> [--resize 64]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def score_video(path: Path, resize: int) -> list[float]:
    cap = cv2.VideoCapture(str(path))
    scores: list[float] = []
    prev_gray = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev_gray is None:
            scores.append(0.0)
        else:
            scores.append(float(np.abs(gray - prev_gray).mean()))
        prev_gray = gray
    cap.release()
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("output_csv", type=Path)
    ap.add_argument("--resize", type=int, default=64,
                     help="downsample frames to resize x resize grayscale before diffing (speed/noise tradeoff)")
    args = ap.parse_args()

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
            scores = score_video(vid, args.resize)
            for idx, s in enumerate(scores):
                w.writerow([ep_id, idx, f"{s:.4f}"])
            if (i + 1) % 10 == 0 or (i + 1) == len(videos):
                print(f"  {i + 1}/{len(videos)} videos, {len(scores)} frames in {ep_id}, "
                      f"{time.time() - t0:.0f}s elapsed", flush=True)

    print(f"wrote {args.output_csv} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
