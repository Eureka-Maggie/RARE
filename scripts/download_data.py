#!/usr/bin/env python3
"""Download the released RARE training and test Parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "EurekaTian/RARE"
DATA_PATTERN = "data/scripts/video_corpus_manual_v1_1b_20260807/**/*.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    destination = args.output_dir.expanduser().resolve()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=[DATA_PATTERN, "release_manifest.json"],
        local_dir=destination,
        force_download=args.force_download,
    )
    print(destination / "data/scripts/video_corpus_manual_v1_1b_20260807")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
