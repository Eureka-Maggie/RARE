#!/usr/bin/env python3
"""Build the public RARE query--rubric dataset from the training Parquet files.

The release preserves the training schema and content while replacing source
video titles and uploader handles with stable, non-identifying group IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

MEDIA = ("2d_animation", "3d_stopmotion", "live_action")
SPLITS = ("train", "test")
EXPECTED_ROWS = {
    "2d_animation/train.parquet": 2125,
    "2d_animation/test.parquet": 237,
    "3d_stopmotion/train.parquet": 1759,
    "3d_stopmotion/test.parquet": 192,
    "live_action/train.parquet": 1509,
    "live_action/test.parquet": 167,
}
PRIVATE_PATTERNS = {
    "absolute_path": re.compile(r"(?:^|\s)/(?:home|root|mnt|tmp|Users|primus[^/]*)/\S+"),
    "internal_service": re.compile(
        "|".join(
            (
                "ali" + "baba(?:-inc)?",
                "ali" + "yun",
                "dash" + "scope",
                r"qu" + r"ark[_-]?" + "mma",
                "oss://",
                "odps://",
                r"internal[_-]?gateway",
            )
        ),
        re.IGNORECASE,
    ),
    "credential": re.compile(r"(?:hf_|sk-)[A-Za-z0-9_-]{15,}", re.IGNORECASE),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
}


def _strings(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)
    elif value is not None:
        yield str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset": "RARE query-rubric annotations",
        "schema_version": 1,
        "total_rows": 0,
        "source_video_groups": 0,
        "files": [],
    }
    seen_query_ids: set[int] = set()
    seen_public_uids: set[str] = set()

    for medium in MEDIA:
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        schema_by_split: dict[str, pa.Schema] = {}
        source_groups: set[str] = set()

        for split in SPLITS:
            source = input_root / medium / f"{split}.parquet"
            if not source.is_file():
                raise FileNotFoundError(source)
            table = pq.read_table(source)
            relative = f"{medium}/{split}.parquet"
            if table.num_rows != EXPECTED_ROWS[relative]:
                raise ValueError(f"{relative}: expected {EXPECTED_ROWS[relative]} rows, found {table.num_rows}")
            rows = table.to_pylist()
            rows_by_split[split] = rows
            schema_by_split[split] = table.schema
            source_groups.update(str(row["extra_info"]["stem"]) for row in rows)

        public_group_ids = {
            source_group: f"{medium}_video_{index:04d}"
            for index, source_group in enumerate(sorted(source_groups), start=1)
        }
        manifest["source_video_groups"] += len(public_group_ids)

        for split in SPLITS:
            released_rows: list[dict[str, Any]] = []
            for row in rows_by_split[split]:
                released = dict(row)
                extra_info = dict(released["extra_info"])
                public_group_id = public_group_ids[str(extra_info["stem"])]
                query_index = int(extra_info["query_idx"])
                public_uid = f"{public_group_id}_q{query_index:02d}"
                extra_info["stem"] = public_group_id
                extra_info["stem_uid"] = public_uid
                released["extra_info"] = extra_info

                query_id = int(released["query_id"])
                if query_id in seen_query_ids:
                    raise ValueError(f"duplicate query_id: {query_id}")
                if public_uid in seen_public_uids:
                    raise ValueError(f"duplicate public query UID: {public_uid}")
                seen_query_ids.add(query_id)
                seen_public_uids.add(public_uid)

                if released.get("answer"):
                    raise ValueError(f"query_id={query_id}: answer must be empty in the annotation release")
                for value in _strings(released):
                    for label, pattern in PRIVATE_PATTERNS.items():
                        if pattern.search(value):
                            raise ValueError(f"query_id={query_id}: {label} marker found")
                released_rows.append(released)

            relative = Path("data/scripts/video_corpus_manual_v1_1b_20260807") / medium / f"{split}.parquet"
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            released_table = pa.Table.from_pylist(released_rows, schema=schema_by_split[split])
            pq.write_table(released_table, destination, compression="zstd", use_dictionary=True)
            manifest["total_rows"] += released_table.num_rows
            manifest["files"].append(
                {
                    "path": relative.as_posix(),
                    "rows": released_table.num_rows,
                    "sha256": _sha256(destination),
                }
            )

    if manifest["total_rows"] != sum(EXPECTED_ROWS.values()):
        raise ValueError(f"unexpected total row count: {manifest['total_rows']}")
    if manifest["source_video_groups"] != 930:
        raise ValueError(f"unexpected source group count: {manifest['source_video_groups']}")

    manifest_path = output_root / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
