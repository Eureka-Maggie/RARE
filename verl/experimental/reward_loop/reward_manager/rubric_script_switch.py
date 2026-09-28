"""Pure helpers for the released joint rubric-evolution method."""

from __future__ import annotations

import copy
import json
import re
from typing import Any


def strict_saturated_ids(rubric: dict[str, Any], observations: list[dict[str, Any]], expected_n: int) -> list[str]:
    """Return criteria that received their maximum score on every valid rollout.

    A failed/missing judge result makes the whole group ineligible.  One group
    of ``expected_n`` judgments is sufficient; there is no warm-up step or
    consecutive-step requirement.
    """
    if len(observations) != expected_n or any(not row.get("judge_valid") for row in observations):
        return []

    maxima = {criterion["id"]: int(criterion["max_points"]) for criterion in rubric.get("criteria", [])}
    if not maxima:
        return []

    saturated: list[str] = []
    for criterion_id, maximum in maxima.items():
        scores = [row.get("scores", {}).get(criterion_id) for row in observations]
        if len(scores) == expected_n and all(score == maximum for score in scores):
            saturated.append(criterion_id)
    return saturated


def render_joint_switch_prompt(
    template: str,
    *,
    query: str,
    current_rubric: dict[str, Any],
    saturated_ids: list[str],
    observations: list[dict[str, Any]],
    global_step: int,
    update_count: int,
    polarity: str,
) -> str:
    """Render all saturated criteria and current rollout evidence in one prompt."""
    criteria_by_id = {row["id"]: row for row in current_rubric["criteria"]}
    saturated = [criteria_by_id[criterion_id] for criterion_id in saturated_ids]
    retained = [row for row in current_rubric["criteria"] if row["id"] not in set(saturated_ids)]
    rollout_blocks = []
    for index, row in enumerate(observations):
        scores = {
            criterion["id"]: row.get("scores", {}).get(criterion["id"]) for criterion in current_rubric["criteria"]
        }
        reasons = {
            criterion_id: reason
            for criterion_id, reason in row.get("reasons", {}).items()
            if isinstance(reason, str) and reason.strip()
        }
        rollout_blocks.append(
            f"### R{index}\n\n"
            f"Current judge scores: {json.dumps(scores, ensure_ascii=False)}\n\n"
            f"Current judge reasons: {json.dumps(reasons, ensure_ascii=False)}\n\n"
            f"{row.get('response', '')}"
        )

    if polarity == "negative":
        polarity_instruction = (
            "每条 update 的 polarity 必须是 negative。目标缺陷没有出现时得满分，"
            "局部出现时逐级扣分，缺陷明显并损害质量时得 0 分。"
        )
    elif polarity == "positive":
        polarity_instruction = (
            "每条 update 的 polarity 必须是 positive。目标质量实现充分时得满分，"
            "实现较弱时逐级降分，正文没有实现时得 0 分。"
        )
    elif polarity == "mix":
        polarity_instruction = (
            "每条 update 可使用 positive 或 negative。若生成两条及以上 update，"
            "结果中必须同时包含两种 polarity；各自的所有分档都保持分数越高质量越好。"
        )
    else:
        raise ValueError(f"unsupported joint switch polarity: {polarity}")

    values = {
        "UPDATE_COUNT": str(update_count),
        "POLARITY": polarity,
        "POLARITY_INSTRUCTION": polarity_instruction,
        "QUERY": query,
        "CURRENT_RUBRIC": json.dumps(current_rubric, ensure_ascii=False, indent=2),
        "SATURATED_CRITERIA": json.dumps(saturated, ensure_ascii=False, indent=2),
        "RETAINED_CRITERIA": json.dumps(retained, ensure_ascii=False, indent=2),
        "CURRENT_ROLLOUTS_WITH_SCORES": "\n\n".join(rollout_blocks),
        "TRIGGER_CONTEXT": json.dumps(
            {
                "type": "CURRENT_GROUP_STRICT_SATURATION",
                "global_step": int(global_step),
                "saturated_ids": saturated_ids,
                "judge_results": len(observations),
            },
            ensure_ascii=False,
            indent=2,
        ),
    }
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _rollout_ids(value: Any, *, field: str, response_count: int) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > response_count
        or any(not isinstance(item, str) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field} must contain unique rollout ids")
    for item in value:
        match = re.fullmatch(r"R(\d+)", item)
        if match is None or not 0 <= int(match.group(1)) < response_count:
            raise ValueError(f"{field} contains an unknown rollout id")
    return list(value)


def validate_joint_switch_output(
    payload: dict[str, Any],
    *,
    current_rubric: dict[str, Any],
    saturated_ids: list[str],
    response_count: int,
    update_count: int,
    polarity: str,
) -> dict[str, Any]:
    """Validate one source-ordered replacement for each selected saturated id."""
    if not isinstance(payload, dict) or set(payload) != {"updates"}:
        raise ValueError("joint writer output must contain only updates")
    updates = payload["updates"]
    if not isinstance(updates, list) or len(updates) != update_count:
        raise ValueError("joint writer update count differs from request")
    criteria_by_id = {row["id"]: row for row in current_rubric.get("criteria", [])}
    if len(criteria_by_id) != len(current_rubric.get("criteria", [])):
        raise ValueError("current rubric has duplicate criterion ids")
    source_order = {criterion_id: index for index, criterion_id in enumerate(saturated_ids)}
    selected_ids = [row.get("id") if isinstance(row, dict) else None for row in updates]
    if (
        len(set(selected_ids)) != update_count
        or any(criterion_id not in source_order for criterion_id in selected_ids)
        or selected_ids != sorted(selected_ids, key=source_order.__getitem__)
    ):
        raise ValueError("joint updates must be a unique source-ordered saturated subset")

    cleaned = []
    seen_rules: set[str] = set()
    observed_polarities: list[str] = []
    for update in updates:
        if set(update) != {"id", "polarity", "diagnosis", "rubric"}:
            raise ValueError("joint update has an invalid shape")
        criterion_id = update["id"]
        update_polarity = update["polarity"]
        if update_polarity not in {"positive", "negative"}:
            raise ValueError("joint update polarity is invalid")
        if polarity in {"positive", "negative"} and update_polarity != polarity:
            raise ValueError("joint update violates configured polarity")
        observed_polarities.append(update_polarity)

        diagnosis = update["diagnosis"]
        expected_diagnosis = {
            "difference",
            "best_rollout_ids",
            "weaker_rollout_ids",
            "learning_direction",
            "nonredundancy",
        }
        if not isinstance(diagnosis, dict) or set(diagnosis) != expected_diagnosis:
            raise ValueError("joint diagnosis has an invalid shape")
        best = _rollout_ids(
            diagnosis["best_rollout_ids"],
            field="best_rollout_ids",
            response_count=response_count,
        )
        weaker = _rollout_ids(
            diagnosis["weaker_rollout_ids"],
            field="weaker_rollout_ids",
            response_count=response_count,
        )
        if set(best) & set(weaker):
            raise ValueError("best and weaker rollout ids overlap")

        rubric = update["rubric"]
        if not isinstance(rubric, dict) or set(rubric) != {
            "id",
            "check",
            "scoring_rule",
            "max_points",
        }:
            raise ValueError("joint replacement rubric has an invalid shape")
        source = criteria_by_id[criterion_id]
        maximum = int(source["max_points"])
        if rubric["id"] != criterion_id or rubric["max_points"] != maximum:
            raise ValueError("joint replacement changed criterion id or max_points")
        check = _require_text(rubric["check"], "joint replacement check")
        scoring_rule = _require_text(rubric["scoring_rule"], "joint replacement scoring_rule")
        if _score_levels(scoring_rule) != set(range(maximum + 1)):
            raise ValueError("joint replacement does not define every score level")
        normalized_rule = _compact_text(check + scoring_rule)
        source_rule = _compact_text(str(source.get("check", "")) + str(source.get("scoring_rule", "")))
        retained_rules = {
            _compact_text(str(row.get("check", "")) + str(row.get("scoring_rule", "")))
            for row in current_rubric.get("criteria", [])
            if row.get("id") != criterion_id
        }
        if normalized_rule == source_rule:
            raise ValueError("joint replacement is unchanged")
        if normalized_rule in retained_rules:
            raise ValueError("joint replacement exactly duplicates another criterion")
        if normalized_rule in seen_rules:
            raise ValueError("joint replacements contain an exact duplicate")
        seen_rules.add(normalized_rule)
        cleaned.append(
            {
                "id": criterion_id,
                "polarity": update_polarity,
                "diagnosis": {
                    "difference": _require_text(diagnosis["difference"], "difference"),
                    "best_rollout_ids": best,
                    "weaker_rollout_ids": weaker,
                    "learning_direction": _require_text(diagnosis["learning_direction"], "learning_direction"),
                    "nonredundancy": _require_text(diagnosis["nonredundancy"], "nonredundancy"),
                },
                "rubric": {
                    "id": criterion_id,
                    "check": check,
                    "scoring_rule": scoring_rule,
                    "max_points": maximum,
                },
            }
        )
    if (
        polarity == "mix"
        and update_count >= 2
        and set(observed_polarities)
        != {
            "positive",
            "negative",
        }
    ):
        raise ValueError("mix requires both positive and negative updates")
    return {"updates": cleaned}


def validate_joint_score_output(
    payload: dict[str, Any],
    *,
    criteria: list[dict[str, Any]],
    response_count: int,
) -> list[dict[str, Any]]:
    """Validate a criterion-major judge score matrix."""
    if not isinstance(payload, dict) or set(payload) != {"criteria"}:
        raise ValueError("joint score output must contain only criteria")
    rows = payload["criteria"]
    if not isinstance(rows, list) or len(rows) != len(criteria):
        raise ValueError("joint score criterion count differs")
    cleaned = []
    for criterion, row in zip(criteria, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"id", "scores"}:
            raise ValueError("joint score criterion row has an invalid shape")
        if row["id"] != criterion["id"]:
            raise ValueError("joint score criterion ids or order changed")
        scores = row["scores"]
        if not isinstance(scores, list) or len(scores) != response_count:
            raise ValueError("joint score rollout count differs")
        clean_scores = []
        maximum = int(criterion["max_points"])
        for index, score_row in enumerate(scores):
            if not isinstance(score_row, dict) or set(score_row) != {
                "rollout_id",
                "score",
                "reason",
            }:
                raise ValueError("joint score rollout row has an invalid shape")
            if score_row["rollout_id"] != f"R{index}":
                raise ValueError("joint score rollout ids or order changed")
            score = score_row["score"]
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= maximum:
                raise ValueError("joint score is outside rubric range")
            clean_scores.append(
                {
                    "rollout_id": f"R{index}",
                    "score": score,
                    "reason": _require_text(score_row["reason"], "joint score reason"),
                }
            )
        cleaned.append({"id": criterion["id"], "scores": clean_scores})
    return cleaned


def replace_criteria(rubric: dict[str, Any], replacements: list[dict[str, Any]]) -> dict[str, Any]:
    """Atomically apply a validated set of unique criterion replacements."""
    result = copy.deepcopy(rubric)
    replacement_by_id = {row["id"]: row for row in replacements}
    if len(replacement_by_id) != len(replacements):
        raise ValueError("joint replacements contain duplicate ids")
    matched = set()
    for index, old in enumerate(result.get("criteria", [])):
        replacement = replacement_by_id.get(old.get("id"))
        if replacement is None:
            continue
        if int(old["max_points"]) != int(replacement["max_points"]):
            raise ValueError("joint replacement max_points changed")
        result["criteria"][index] = copy.deepcopy(replacement)
        matched.add(old["id"])
    if matched != set(replacement_by_id):
        raise ValueError("joint replacement target is absent from rubric")
    return result


def _score_levels(scoring_rule: str) -> set[int]:
    return {int(value) for value in re.findall(r"(?<!\d)(\d+)\s*=", scoring_rule)}


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value)
