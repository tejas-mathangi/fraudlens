#!/usr/bin/env python3
"""Fetch the Elliptic Bitcoin dataset into ``data/elliptic_bitcoin_dataset/``.

The dataset is ~697 MB and Kaggle-licensed, so it is not redistributed in this
repository. This script uses ``kagglehub``, which needs Kaggle credentials either in
``~/.kaggle/kaggle.json`` or as ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``.

If you would rather not install ``kagglehub``, download the dataset manually from
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set and unpack the three
CSVs into the target directory printed below.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudlens.config import PATHS  # noqa: E402

DATASET = "ellipticco/elliptic-data-set"
REQUIRED = (
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
)


def main() -> int:
    target = PATHS.data_dir
    target.mkdir(parents=True, exist_ok=True)

    if PATHS.dataset_available():
        print(f"Dataset already present in {target}")
        return 0

    try:
        import kagglehub
    except ImportError:
        print(
            "kagglehub is not installed.\n\n"
            "  pip install kagglehub\n\n"
            f"Or download {DATASET} manually and place these files in {target}:\n"
            + "\n".join(f"  - {name}" for name in REQUIRED),
            file=sys.stderr,
        )
        return 1

    print(f"Downloading {DATASET} (~697 MB)...")
    source = Path(kagglehub.dataset_download(DATASET))

    # The archive nests the CSVs one or two levels deep depending on the release.
    for name in REQUIRED:
        matches = list(source.rglob(name))
        if not matches:
            print(f"Could not find {name} in {source}", file=sys.stderr)
            return 1
        dest = target / name
        if not dest.exists():
            shutil.copy2(matches[0], dest)
            print(f"  {name} -> {dest}")

    if not PATHS.dataset_available():
        print("Download finished but files are still missing.", file=sys.stderr)
        return 1

    print("\nDone. Next: fraudlens eda && fraudlens train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
