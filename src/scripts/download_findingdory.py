"""Download the FindingDory metadata (question/answer parquet files + episode-id
mapping) from Hugging Face into data/findingdory/.

This fetches ONLY the metadata parquet files (~15 MB total for train+validation),
not the 32.5 GB videos.zip. The parquet files are enough to run Milestone 1
(index-space selection policies scored against the oracle frame-index answers);
raw video frames are only needed for later milestones that use actual pixel
content (e.g. a surprise/novelty signal from frame embeddings).

Usage:
    python3 src/scripts/download_findingdory.py            # validation split only (default, ~1.4 MB)
    python3 src/scripts/download_findingdory.py --train    # also fetch the train split (~13 MB)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

REPO = "yali30/findingdory"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "findingdory"

FILES_DEFAULT = [
    "data/validation-00000-of-00001.parquet",
    "episode_id_mappings.json",
]
FILES_TRAIN = ["data/train-00000-of-00001.parquet"]


def fetch(remote_path: str, out_dir: Path) -> Path:
    out_path = out_dir / Path(remote_path).name
    if out_path.exists():
        print(f"  already present: {out_path}")
        return out_path
    url = f"{BASE_URL}/{remote_path}"
    print(f"  fetching {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    print(f"  wrote {out_path} ({len(resp.content) / 1e6:.2f} MB)")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true", help="also fetch the train-split parquet (~13 MB)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = list(FILES_DEFAULT) + (FILES_TRAIN if args.train else [])
    print(f"Downloading FindingDory metadata into {DATA_DIR}")
    for f in files:
        fetch(f, DATA_DIR)
    print("Done. Note: videos.zip (32.5 GB) was NOT downloaded; it is not needed for Milestone 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
