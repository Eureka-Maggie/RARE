"""Fail-closed validator for the released Overall-v9 validation judge."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

METRICS = [
    "query_requirement_coverage",
    "query_grounded_elaboration",
    "tone_style_alignment",
    "causal_temporal_logic",
    "character_motivation_behavior",
    "cross_scene_state_continuity",
    "visual_audio_observability",
    "shot_feasibility_atomicity",
    "pacing_shot_utility",
    "engagement_progression",
    "setup_payoff_ending",
    "distinctiveness_resonance",
]

CATEGORIES = {
    "query_fulfillment": METRICS[0:3],
    "narrative_coherence": METRICS[3:6],
    "cinematic_executability": METRICS[6:9],
    "dramatic_effectiveness": METRICS[9:12],
}

DECISION_SCORE = {"pass": 2, "partial": 1, "fail": 0}

CHECK_IDS = {
    "query_requirement_coverage": [
        "required_content_realized",
        "requested_trajectory_result",
        "formal_constraints_satisfied",
    ],
    "query_grounded_elaboration": [
        "major_additions_traceable",
        "no_contradictory_replacement",
        "focus_preserved",
    ],
    "tone_style_alignment": [
        "main_scenes_match_style",
        "turn_climax_match_style",
        "ending_match_style",
    ],
    "causal_temporal_logic": [
        "triggers_established",
        "actions_have_consequences",
        "chronology_reveals_consistent",
    ],
    "character_motivation_behavior": [
        "goals_knowledge_pressure_established",
        "choices_consistent_with_known_information",
        "change_has_behavioral_cause",
    ],
    "cross_scene_state_continuity": [
        "entity_object_state_continuous",
        "knowledge_resource_state_continuous",
        "space_time_transition_unambiguous",
    ],
    "visual_audio_observability": [
        "core_conflict_rule_observable",
        "motivation_turn_observable",
        "cause_result_observable",
    ],
    "shot_feasibility_atomicity": [
        "complex_sequence_executable",
        "transitions_effects_consistent",
        "instructions_complete_nonconflicting",
    ],
    "pacing_shot_utility": [
        "all_blocks_add_nonoverlapping_function",
        "no_redundant_explanation_pattern",
        "key_beats_sufficiently_developed",
    ],
    "engagement_progression": [
        "opening_focus_established",
        "middle_pressure_changes",
        "turn_changes_action",
    ],
    "setup_payoff_ending": [
        "significant_setups_transformed",
        "ending_causally_earned",
        "ending_preserves_inference",
    ],
    "distinctiveness_resonance": [
        "central_mechanism_causally_specific",
        "character_action_combination_specific",
        "final_image_leaves_aftereffect",
    ],
}

ZERO_THRESHOLDS = {
    "query_requirement_coverage": {
        "identity_requirement_missing_or_replaced",
        "incomplete_stage_prevents_requested_result",
    },
    "query_grounded_elaboration": {
        "majority_additions_off_query_or_contradictory",
    },
    "tone_style_alignment": {
        "dominant_experience_opposes_or_lacks_requested_style",
    },
    "causal_temporal_logic": {
        "core_causal_chain_absent_or_contradictory",
    },
    "character_motivation_behavior": {
        "central_actions_unmotivated_or_self_contradictory",
    },
    "cross_scene_state_continuity": {
        "actual_state_contradictions_invalidate_core_result",
    },
    "visual_audio_observability": {
        "core_story_depends_on_exposition_without_playable_support",
    },
    "shot_feasibility_atomicity": {
        "multiple_conflicting_or_unexecutable_core_sequences",
    },
    "pacing_shot_utility": {
        "multiple_major_gaps_or_systemic_redundancy",
        "incomplete_stage_omits_climax_or_ending",
    },
    "engagement_progression": {
        "no_progression_engine_or_consequential_turn",
    },
    "setup_payoff_ending": {
        "ending_missing_arbitrary_or_wholly_unearned",
        "incomplete_stage_has_no_written_ending",
    },
    "distinctiveness_resonance": {
        "generic_template_without_causally_specific_choice",
    },
}


def _clean_string_list(value: Any, label: str, *, max_items: int | None = None) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if max_items is not None and len(value) > max_items:
        raise ValueError(f"{label} permits at most {max_items} items")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{label} must contain only non-empty strings")
    return [item.strip() for item in value]


def _clean_problem_list(value: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    cleaned: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            raw = item.strip()
            parts = re.split(r"[：:]", raw, maxsplit=1)
            location = parts[0].strip()
            reason = parts[1].strip() if len(parts) == 2 else raw
            cleaned.append({"location": location, "reason": reason})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{label} entries must be objects")
        location = item.get("location")
        reason = item.get("reason")
        if not isinstance(location, str) or not location.strip():
            raise ValueError(f"{label} entries require a non-empty location")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{label} entries require a non-empty reason")
        cleaned.append({"location": location.strip(), "reason": reason.strip()})
    return cleaned


def _clean_integrity_evidence(value: Any) -> list[Any]:
    """Accept concise strings or structured locators for the non-scored stage audit.

    Exact substring verification remains mandatory for all 36 scored checks.
    Integrity evidence may summarize whole-document completeness and is therefore
    kept separate from metric evidence.
    """
    if not isinstance(value, list) or not value:
        raise ValueError("input_integrity.evidence must be a non-empty list")
    cleaned: list[Any] = []
    for index, item in enumerate(value):
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
            continue
        if not isinstance(item, dict):
            raise ValueError(f"input_integrity.evidence[{index}] must be a string or evidence object")
        location = item.get("location")
        quote = item.get("quote")
        interpretation = item.get("interpretation")
        if not all(isinstance(found, str) and found.strip() for found in (location, quote, interpretation)):
            raise ValueError(f"input_integrity.evidence[{index}] object requires location, quote, and interpretation")
        cleaned.append(
            {
                "location": location.strip(),
                "quote": quote.strip(),
                "interpretation": interpretation.strip(),
            }
        )
    return cleaned


def _clean_v8_defects(value: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    cleaned: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{label} entries must be objects")
        location = item.get("location")
        reason = item.get("reason")
        severity = item.get("severity")
        if not isinstance(location, str) or not location.strip():
            raise ValueError(f"{label} entries require a non-empty location")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{label} entries require a non-empty reason")
        if severity not in {"major", "critical"}:
            raise ValueError(f"{label} invalid severity: {severity!r}")
        cleaned.append(
            {
                "location": location.strip(),
                "reason": reason.strip(),
                "severity": severity,
            }
        )
    return cleaned


def _normalize_quote(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(char.lower() for char in normalized if char.isalnum())


def _clean_evidence_items(
    value: Any,
    label: str,
    candidate_text: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    normalized_candidate = _normalize_quote(candidate_text)
    cleaned: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        location = item.get("location")
        quote = item.get("quote")
        interpretation = item.get("interpretation")
        if not isinstance(location, str) or not location.strip():
            raise ValueError(f"{label}[{index}].location must be non-empty")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"{label}[{index}].quote must be non-empty")
        if not isinstance(interpretation, str) or not interpretation.strip():
            raise ValueError(f"{label}[{index}].interpretation must be non-empty")
        normalized_quote = _normalize_quote(quote)
        fragments = [fragment.strip() for fragment in re.split(r"(?:\.{2,}|…+)", quote) if fragment.strip()]
        if len(fragments) >= 2:
            normalized_fragments = [_normalize_quote(fragment) for fragment in fragments]
            cursor = 0
            positions: list[int] = []
            if all(normalized_fragments):
                for fragment in normalized_fragments:
                    position = normalized_candidate.find(fragment, cursor)
                    if position < 0:
                        positions = []
                        break
                    positions.append(position)
                    cursor = position + len(fragment)
            if not positions:
                raise ValueError(f"{label}[{index}].quote is not a verifiable candidate substring: {quote!r}")
            for fragment_index, fragment in enumerate(fragments, start=1):
                cleaned.append(
                    {
                        "location": location.strip(),
                        "quote": fragment,
                        "interpretation": (
                            f"{interpretation.strip()} [canonicalized excerpt {fragment_index}/{len(fragments)}]"
                        ),
                    }
                )
            continue
        if normalized_quote not in normalized_candidate:
            raise ValueError(f"{label}[{index}].quote is not a verifiable candidate substring: {quote!r}")
        if len(normalized_quote) < 1:
            raise ValueError(f"{label}[{index}].quote is too short to verify")
        cleaned.append(
            {
                "location": location.strip(),
                "quote": quote.strip(),
                "interpretation": interpretation.strip(),
            }
        )
    return cleaned


def _clean_adversarial_probe(
    value: Any,
    label: str,
    candidate_text: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    risk = value.get("risk")
    reason = value.get("reason")
    verdict = value.get("verdict")
    if not isinstance(risk, str) or not risk.strip():
        raise ValueError(f"{label}.risk must be non-empty")
    if verdict not in {"material", "not_material"}:
        raise ValueError(f"{label}.verdict must be material or not_material")
    evidence = _clean_evidence_items(value.get("evidence"), f"{label}.evidence", candidate_text)
    if not evidence:
        raise ValueError(f"{label}.evidence must contain at least one item")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{label}.reason must be non-empty")
    return {
        "risk": risk.strip(),
        "evidence": evidence,
        "verdict": verdict,
        "reason": reason.strip(),
    }


def validate_v9(obj: dict[str, Any], candidate_text: str) -> dict[str, Any]:
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        raise ValueError("v9 validation requires the non-empty candidate text")

    expected_top_order = [
        "rubric_version",
        "query_analysis",
        "input_integrity",
        "evaluation_stage",
        "input_warnings",
        "audits",
        "metrics",
    ]
    present_order = [key for key in obj if key in expected_top_order]
    if present_order != expected_top_order:
        raise ValueError(
            "v9 top-level fields must be emitted in evidence-audit-first order: " + ", ".join(expected_top_order)
        )

    query_analysis = obj.get("query_analysis")
    if not isinstance(query_analysis, dict) or set(query_analysis) != {
        "required_content",
        "formal_constraints",
        "style_preferences",
    }:
        raise ValueError("v9 query_analysis must contain exactly three required lists")
    clean_query_analysis = {
        key: _clean_string_list(query_analysis[key], f"query_analysis.{key}")
        for key in ("required_content", "formal_constraints", "style_preferences")
    }
    if not clean_query_analysis["required_content"]:
        raise ValueError("v9 query_analysis.required_content must not be empty")

    integrity = obj.get("input_integrity")
    if not isinstance(integrity, dict):
        raise ValueError("v9 input_integrity must be an object")
    integrity_status = integrity.get("status")
    if integrity_status not in {"complete", "incomplete", "uncertain"}:
        raise ValueError(f"v9 invalid input_integrity.status: {integrity_status!r}")
    evaluation_stage = obj.get("evaluation_stage")
    if evaluation_stage != integrity_status:
        raise ValueError("v9 evaluation_stage must exactly match input_integrity.status")
    integrity_evidence = _clean_integrity_evidence(integrity.get("evidence"))
    integrity_defects = _clean_problem_list(integrity.get("defects"), "input_integrity.defects")
    if integrity_status == "complete" and integrity_defects:
        raise ValueError("v9 complete input requires empty integrity defects")
    if integrity_status == "incomplete" and not integrity_defects:
        raise ValueError("v9 incomplete input requires a locatable integrity defect")
    clean_integrity = {
        "status": integrity_status,
        "evidence": integrity_evidence,
        "defects": integrity_defects,
    }

    input_warnings = obj.get("input_warnings")
    if not isinstance(input_warnings, list) or not all(isinstance(item, str) for item in input_warnings):
        raise ValueError("v9 input_warnings must be a string list")

    audits = obj.get("audits")
    if not isinstance(audits, dict) or set(audits) != set(METRICS):
        raise ValueError("v9 audits must contain exactly the 12 required slots")
    metrics = obj.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise ValueError("v9 metrics must contain exactly the 12 required slots")

    clean_audits: dict[str, Any] = {}
    clean_metrics: dict[str, Any] = {}
    for name in METRICS:
        audit = audits[name]
        if not isinstance(audit, dict):
            raise ValueError(f"v9 audit {name} is not an object")
        raw_checks = audit.get("checks")
        if not isinstance(raw_checks, list) or len(raw_checks) != 3:
            raise ValueError(f"v9 {name}.checks must contain exactly three checks")
        clean_checks: list[dict[str, Any]] = []
        for index, check in enumerate(raw_checks):
            if not isinstance(check, dict):
                raise ValueError(f"v9 {name}.checks entries must be objects")
            check_id = check.get("check_id")
            expected_check_id = CHECK_IDS[name][index]
            if check_id != expected_check_id:
                raise ValueError(f"v9 {name}.checks[{index}] requires {expected_check_id!r}, got {check_id!r}")
            status = check.get("status")
            if status not in {"pass", "fail"}:
                raise ValueError(f"v9 invalid check status for {name}.{check_id}: {status!r}")
            supporting = _clean_evidence_items(
                check.get("supporting_evidence"),
                f"audits.{name}.{check_id}.supporting_evidence",
                candidate_text,
            )
            counter = _clean_evidence_items(
                check.get("counterevidence"),
                f"audits.{name}.{check_id}.counterevidence",
                candidate_text,
            )
            if status == "pass" and (not supporting or counter):
                raise ValueError(f"v9 pass check requires support and no counterevidence for {name}.{check_id}")
            if status == "fail" and not counter:
                raise ValueError(f"v9 failed check requires counterevidence for {name}.{check_id}")
            clean_checks.append(
                {
                    "check_id": check_id,
                    "status": status,
                    "supporting_evidence": supporting,
                    "counterevidence": counter,
                }
            )

        adversarial_probe = _clean_adversarial_probe(
            audit.get("adversarial_probe"),
            f"audits.{name}.adversarial_probe",
            candidate_text,
        )
        critical_turn_dependency: dict[str, str] | None = None
        if name == "causal_temporal_logic":
            raw_dependency = audit.get("critical_turn_dependency")
            if not isinstance(raw_dependency, dict):
                raise ValueError("v9 causal_temporal_logic requires critical_turn_dependency")
            classification = raw_dependency.get("classification")
            dependency_reason = raw_dependency.get("reason")
            if classification not in {
                "established",
                "reasonable_inference",
                "coincidence",
                "missing",
            }:
                raise ValueError(f"v9 invalid critical turn dependency classification: {classification!r}")
            if not isinstance(dependency_reason, str) or not dependency_reason.strip():
                raise ValueError("v9 critical_turn_dependency.reason must be non-empty")
            critical_turn_dependency = {
                "classification": classification,
                "reason": dependency_reason.strip(),
            }
        defects = _clean_v8_defects(audit.get("defects"), f"audits.{name}.defects")
        decision = audit.get("decision")
        if decision not in DECISION_SCORE:
            raise ValueError(f"v9 invalid decision for {name}: {decision!r}")
        zero_threshold = audit.get("zero_threshold_hit")
        failed_checks = [check for check in clean_checks if check["status"] == "fail"]
        critical_defects = [defect for defect in defects if defect["severity"] == "critical"]
        if decision == "pass":
            if failed_checks or defects:
                raise ValueError(f"v9 pass requires all checks pass and no defects for {name}")
            if zero_threshold is not None:
                raise ValueError(f"v9 pass requires zero_threshold_hit=null for {name}")
            if adversarial_probe["verdict"] != "not_material":
                raise ValueError(f"v9 pass requires a not_material adversarial probe for {name}")
            if name == "causal_temporal_logic" and critical_turn_dependency["classification"] in {
                "coincidence",
                "missing",
            }:
                raise ValueError("v9 causal pass forbids coincidence/missing critical turn dependency")
        elif decision == "partial":
            if not failed_checks or not defects or critical_defects:
                raise ValueError(
                    f"v9 partial requires failed check(s), major defect(s), and no critical defect for {name}"
                )
            if zero_threshold is not None:
                raise ValueError(f"v9 partial requires zero_threshold_hit=null for {name}")
            if adversarial_probe["verdict"] != "material":
                raise ValueError(f"v9 partial requires a material adversarial probe for {name}")
        else:
            if not failed_checks or not critical_defects:
                raise ValueError(f"v9 fail requires failed check(s) and a critical defect for {name}")
            if zero_threshold not in ZERO_THRESHOLDS[name]:
                raise ValueError(f"v9 fail for {name} requires an allowed zero_threshold_hit; got {zero_threshold!r}")
            if adversarial_probe["verdict"] != "material":
                raise ValueError(f"v9 fail requires a material adversarial probe for {name}")

        clean_audit: dict[str, Any] = {
            "checks": clean_checks,
            "adversarial_probe": adversarial_probe,
            "defects": defects,
            "zero_threshold_hit": zero_threshold,
            "decision": decision,
        }
        if critical_turn_dependency is not None:
            clean_audit["critical_turn_dependency"] = critical_turn_dependency
        if name == "pacing_shot_utility":
            progression = _clean_string_list(
                audit.get("functional_progression"),
                "audits.pacing_shot_utility.functional_progression",
                max_items=8,
            )
            raw_blocks = audit.get("block_audit")
            if not isinstance(raw_blocks, list) or not raw_blocks:
                raise ValueError("v9 pacing block_audit must be a non-empty list")
            block_audit: list[dict[str, Any]] = []
            for block in raw_blocks:
                if not isinstance(block, dict):
                    raise ValueError("v9 pacing block_audit entries must be objects")
                location = block.get("location")
                new_function = block.get("new_function")
                overlap = block.get("overlap_with")
                verdict = block.get("verdict")
                if not isinstance(location, str) or not location.strip():
                    raise ValueError("v9 pacing block location must be non-empty")
                if not isinstance(new_function, str) or not new_function.strip():
                    raise ValueError("v9 pacing block new_function must be non-empty")
                if overlap is not None and (not isinstance(overlap, str) or not overlap.strip()):
                    raise ValueError("v9 pacing overlap_with must be null or non-empty")
                if verdict not in {"essential", "mergeable", "removable"}:
                    raise ValueError(f"v9 pacing invalid verdict: {verdict!r}")
                block_audit.append(
                    {
                        "location": location.strip(),
                        "new_function": new_function.strip(),
                        "overlap_with": overlap.strip() if isinstance(overlap, str) else None,
                        "verdict": verdict,
                    }
                )
            removable = _clean_problem_list(
                audit.get("removable_blocks"),
                "audits.pacing_shot_utility.removable_blocks",
            )
            underwritten = _clean_problem_list(
                audit.get("underwritten_blocks"),
                "audits.pacing_shot_utility.underwritten_blocks",
            )
            if decision in {"partial", "fail"} and not (removable or underwritten):
                recovered_problems = [
                    {"location": defect["location"], "reason": defect["reason"]} for defect in defects
                ]
                failed_check_ids = {check["check_id"] for check in failed_checks}
                if "key_beats_sufficiently_developed" in failed_check_ids:
                    underwritten = recovered_problems
                else:
                    removable = recovered_problems
            if decision == "pass":
                if any(block["verdict"] != "essential" for block in block_audit):
                    raise ValueError("v9 pacing pass requires every block to be essential")
                if removable or underwritten:
                    raise ValueError("v9 pacing pass requires empty problem lists")
            elif not (removable or underwritten):
                raise ValueError("v9 pacing partial/fail requires a locatable problem block")
            if not progression and decision != "fail":
                raise ValueError("v9 empty pacing progression is allowed only for fail")
            clean_audit.update(
                {
                    "functional_progression": progression,
                    "block_audit": block_audit,
                    "removable_blocks": removable,
                    "underwritten_blocks": underwritten,
                }
            )
        clean_audits[name] = clean_audit

        item = metrics[name]
        if not isinstance(item, dict):
            raise ValueError(f"v9 metric {name} is not an object")
        score = item.get("score")
        expected_score = DECISION_SCORE[decision]
        if isinstance(score, bool) or score != expected_score:
            raise ValueError(
                f"v9 decision-score mismatch for {name}: {decision!r} requires {expected_score}, got {score!r}"
            )
        if item.get("max_score") != 2:
            raise ValueError(f"v9 max_score must be 2 for {name}")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"v9 metric {name} requires a non-empty reason")
        clean_metrics[name] = {"score": score, "max_score": 2, "reason": reason.strip()}

    if integrity_status == "incomplete":
        required_fails = {
            "query_requirement_coverage": "incomplete_stage_prevents_requested_result",
            "pacing_shot_utility": "incomplete_stage_omits_climax_or_ending",
            "setup_payoff_ending": "incomplete_stage_has_no_written_ending",
        }
        for name, threshold in required_fails.items():
            audit = clean_audits[name]
            if audit["decision"] != "fail" or audit["zero_threshold_hit"] != threshold:
                raise ValueError(f"v9 incomplete input requires {name}=fail with zero threshold {threshold!r}")
        for name in {
            "tone_style_alignment",
            "shot_feasibility_atomicity",
            "engagement_progression",
            "distinctiveness_resonance",
        }:
            if clean_audits[name]["decision"] == "pass":
                raise ValueError(f"v9 incomplete input forbids {name}=pass")

    category_scores = {
        category: sum(clean_metrics[name]["score"] for name in names) for category, names in CATEGORIES.items()
    }
    category_max_scores = {category: 6 for category in CATEGORIES}
    category_score_100 = {category: round(score / 6 * 100, 2) for category, score in category_scores.items()}
    total = sum(item["score"] for item in clean_metrics.values())
    reported = {
        "category_scores": obj.get("category_scores"),
        "total_score": obj.get("total_score"),
        "total_max": obj.get("total_max"),
        "overall_score_100": obj.get("overall_score_100"),
    }
    return {
        "rubric_version": "microfilm_overall_12_v9",
        "query_analysis": clean_query_analysis,
        "input_integrity": clean_integrity,
        "evaluation_stage": evaluation_stage,
        "input_warnings": input_warnings,
        "audits": clean_audits,
        "metrics": clean_metrics,
        "active_metric_count": 12,
        "category_scores": category_scores,
        "category_max_scores": category_max_scores,
        "category_score_100": category_score_100,
        "total_score": total,
        "total_max": 24,
        "overall_score_100": round(total / 24 * 100, 2),
        "judge_reported_arithmetic": reported,
    }


def validate_and_canonicalize(obj: dict[str, Any], candidate_text: str) -> dict[str, Any]:
    rubric_version = obj.get("rubric_version")
    if rubric_version != "microfilm_overall_12_v9":
        raise ValueError(f"validator does not accept rubric version {rubric_version!r}")
    return validate_v9(obj, candidate_text)
