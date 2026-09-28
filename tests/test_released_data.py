import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/scripts/video_corpus_manual_v1_1b_20260807"
EXPECTED_ROWS = {
    "2d_animation/train.parquet": 2125,
    "2d_animation/test.parquet": 237,
    "3d_stopmotion/train.parquet": 1759,
    "3d_stopmotion/test.parquet": 192,
    "live_action/train.parquet": 1509,
    "live_action/test.parquet": 167,
}
REQUIRED_COLUMNS = {"prompt", "extra_info", "data_source", "reward_model"}
PRIVATE_MARKERS = re.compile(
    "|".join(
        (
            "ali" + "baba",
            "ali" + "babacloud",
            "ali" + "yun",
            "dash" + "scope",
            r"llm[ _-]?" + "chat",
            "kuai" + "pao",
            "blank" + "api",
            r"qu" + r"ark[_-]?" + "mma",
            "pri" + "mus",
            r"internal[_-]?gateway",
        )
    ),
    re.IGNORECASE,
)
PUBLIC_STEM = re.compile(r"(?:2d_animation|3d_stopmotion|live_action)_video_\d{4}")
PUBLIC_STEM_UID = re.compile(r"(?:2d_animation|3d_stopmotion|live_action)_video_\d{4}_q\d{2}")


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


def test_release_manifest_matches_downloaded_files():
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_rows"] == 5989
    assert manifest["source_video_groups"] == 930
    assert len(manifest["files"]) == 6

    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file()
        assert pq.read_metadata(path).num_rows == item["rows"]
        assert _sha256(path) == item["sha256"]


def test_released_parquet_files_are_complete_and_sanitized():
    found = {str(path.relative_to(DATA_ROOT)): path for path in DATA_ROOT.rglob("*.parquet")}
    assert set(found) == set(EXPECTED_ROWS)
    query_ids = set()
    source_groups = set()
    public_uids = set()

    for relative_path, expected_rows in EXPECTED_ROWS.items():
        table = pq.read_table(found[relative_path])
        assert table.num_rows == expected_rows
        assert REQUIRED_COLUMNS <= set(table.column_names)

        for row in table.to_pylist():
            prompt = row["prompt"]
            assert isinstance(prompt, list) and prompt
            assert all(
                isinstance(message, dict)
                and isinstance(message.get("role"), str)
                and isinstance(message.get("content"), str)
                for message in prompt
            )
            assert isinstance(row["extra_info"], dict)
            assert row["extra_info"].get("criteria")
            assert row["answer"] == ""

            query_id = row["query_id"]
            stem = row["extra_info"].get("stem", "")
            stem_uid = row["extra_info"].get("stem_uid", "")
            assert query_id not in query_ids
            assert stem_uid not in public_uids
            assert PUBLIC_STEM.fullmatch(stem)
            assert PUBLIC_STEM_UID.fullmatch(stem_uid)
            query_ids.add(query_id)
            source_groups.add(stem)
            public_uids.add(stem_uid)
            assert not any(PRIVATE_MARKERS.search(value) for value in _strings(row))

    assert len(query_ids) == 5989
    assert len(source_groups) == 930
    assert len(public_uids) == 5989
