"""Download NASA C-MAPSS FD001-FD004 dataset files into data/raw/."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
FILES = [
    "readme.txt",
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
    "train_FD002.txt",
    "test_FD002.txt",
    "RUL_FD002.txt",
    "train_FD003.txt",
    "test_FD003.txt",
    "RUL_FD003.txt",
    "train_FD004.txt",
    "test_FD004.txt",
    "RUL_FD004.txt",
]


def download_cmapss(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        dest = dest_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {fname} already exists ({dest.stat().st_size} bytes)")
            continue
        url = f"{BASE_URL}/{fname}"
        print(f"[download] {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"[ok] {fname}: {dest.stat().st_size} bytes")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    download_cmapss(root / "data" / "raw")
