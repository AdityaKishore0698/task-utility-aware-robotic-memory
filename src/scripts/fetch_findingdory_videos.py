"""Fetch only specific episode videos out of FindingDory's videos.zip (32.5 GB)
via HTTP range requests, without downloading the whole archive.

How: a ZIP's central directory (the file listing + per-entry offsets/sizes) sits
at the END of the file, so we can read just that (a couple of small range
requests) to learn where each entry lives, then issue exactly ONE range request
per episode covering that entry's local header + compressed bytes, and inflate
locally if needed. This is the standard technique behind tools like `remotezip`,
reimplemented here directly on `requests` + stdlib `zipfile`/`zlib` to avoid an
extra dependency for ~100 files.

Caveats (be honest about what this does and does not verify):
  - This has been tested against yali30/findingdory's videos.zip specifically.
    It assumes a standard (non-Zip64) end-of-central-directory record, which is
    an assumption, not something independently confirmed against this archive's
    internals ahead of time -- the script will raise clearly if that assumption
    is wrong rather than silently producing a corrupt file.
  - Only CRC32 (built into the zip format) verifies each extracted file; we do
    not have an independent checksum from the dataset authors to compare against.

Usage:
    python3 src/scripts/fetch_findingdory_videos.py --split validation
    python3 src/scripts/fetch_findingdory_videos.py --split validation --episodes ep_1,ep_2
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import requests

REPO = "yali30/findingdory"
ZIP_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/videos.zip"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "findingdory" / "videos"

LOCAL_HEADER_FIXED_SIZE = 30  # bytes, per the ZIP local file header spec


class RangeReader:
    """A minimal seekable/readable file-like object backed by HTTP range
    requests, just enough for zipfile to read the central directory from the
    tail of a remote file without downloading the rest."""

    def __init__(self, url: str, session: requests.Session):
        self.url = url
        self.session = session
        head = session.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            end = self.size - 1
        else:
            end = min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        r = self.session.get(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60
        )
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data

    def seekable(self) -> bool:
        return True


def fetch_entry(session: requests.Session, url: str, info: ZipInfo, out_path: Path) -> None:
    """Fetch one zip entry's bytes with a single range request and write the
    decompressed (if needed) content to out_path."""
    start = info.header_offset
    # Local header has a fixed 30-byte part, then filename + extra field,
    # whose lengths we don't know exactly until we've read the fixed part --
    # over-fetch a generous slack (filenames/extras are always tiny) so this
    # stays a single request.
    slack = 512
    end = start + LOCAL_HEADER_FIXED_SIZE + slack + info.compress_size
    r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    blob = r.content

    # Local file header (30-byte fixed part): signature, version, flags,
    # compression, mod_time, mod_date, crc32, compressed_size,
    # uncompressed_size, filename_len, extra_len.
    sig, _, _, _, _, _, _, _, _, fname_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", blob[:30]
    )
    if sig != 0x04034B50:
        raise ValueError(f"{info.filename}: bad local file header signature {sig:#x}")
    data_start = LOCAL_HEADER_FIXED_SIZE + fname_len + extra_len
    compressed = blob[data_start : data_start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise ValueError(
            f"{info.filename}: expected {info.compress_size} compressed bytes, "
            f"got {len(compressed)} (increase `slack` if filenames/extras are unusually long)"
        )

    if info.compress_type == 0:  # stored, no compression
        payload = compressed
    elif info.compress_type == 8:  # deflate
        payload = zlib.decompressobj(-15).decompress(compressed)
    else:
        raise ValueError(f"{info.filename}: unsupported compress_type {info.compress_type}")

    if len(payload) != info.file_size:
        raise ValueError(
            f"{info.filename}: decompressed to {len(payload)} bytes, expected {info.file_size}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    ap.add_argument(
        "--episodes",
        default=None,
        help="comma-separated ep_id list to fetch (default: all episodes in the split)",
    )
    args = ap.parse_args()

    session = requests.Session()
    print(f"reading central directory of {ZIP_URL} (no full download)...")
    t0 = time.time()
    rr = RangeReader(ZIP_URL, session)
    print(f"  archive size: {rr.size / 1e9:.2f} GB")
    zf = ZipFile(rr)  # only reads the central directory (tail of the file)
    print(f"  central directory read in {time.time() - t0:.1f}s, {len(zf.namelist())} entries total")

    prefix = f"videos/{'val' if args.split == 'validation' else 'train'}/"
    wanted = set(args.episodes.split(",")) if args.episodes else None

    entries = []
    for info in zf.infolist():
        if not info.filename.startswith(prefix) or not info.filename.endswith(".mp4"):
            continue
        ep_id = Path(info.filename).stem
        if wanted is not None and ep_id not in wanted:
            continue
        entries.append((ep_id, info))

    if not entries:
        print("no matching entries found", file=sys.stderr)
        return 1

    total_bytes = sum(info.file_size for _, info in entries)
    print(f"fetching {len(entries)} episode videos ({total_bytes / 1e9:.2f} GB uncompressed) "
          f"into {OUT_DIR / args.split}/ ...")

    t1 = time.time()
    done_bytes = 0
    for i, (ep_id, info) in enumerate(entries):
        out_path = OUT_DIR / args.split / f"{ep_id}.mp4"
        if out_path.exists() and out_path.stat().st_size == info.file_size:
            done_bytes += info.file_size
            continue
        fetch_entry(session, ZIP_URL, info, out_path)
        done_bytes += info.file_size
        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            elapsed = time.time() - t1
            rate = done_bytes / elapsed / 1e6 if elapsed > 0 else 0
            print(f"  {i + 1}/{len(entries)} episodes, {done_bytes / 1e9:.2f} GB, "
                  f"{rate:.1f} MB/s avg, {elapsed:.0f}s elapsed")

    print(f"done in {time.time() - t1:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
