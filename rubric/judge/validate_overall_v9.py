"""Adapter for the canonical Overall-v9 validator used by validation rewards."""

from __future__ import annotations

import math
from typing import Any

from rubric.judge.overall_v9_validator import validate_and_canonicalize


def extract_validated_v9_score(payload: dict[str, Any], candidate_text: str) -> float:
    """Return the canonical Overall-v9 score on a 0--100 scale."""
    if payload.get("rubric_version") != "microfilm_overall_12_v9":
        raise ValueError(f"expected microfilm_overall_12_v9, got {payload.get('rubric_version')!r}")
    canonical = validate_and_canonicalize(payload, candidate_text)
    score = canonical.get("overall_score_100")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"v9 overall_score_100 is not numeric: {score!r}")
    score = float(score)
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError(f"v9 overall_score_100 outside [0, 100]: {score}")
    return score
