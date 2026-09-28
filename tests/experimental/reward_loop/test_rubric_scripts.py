import asyncio
import json

import numpy as np
import pytest
import torch

from rubric.judge import validate_overall_v9
from verl.experimental.reward_loop.reward_manager import rubric_scripts
from verl.trainer.ppo.ray_trainer import _prepare_grpo_judge_inputs


class _ImmediateLoop:
    async def run_in_executor(self, _executor, fn):
        return fn()


def _train_criteria() -> list[dict]:
    return [
        {
            "id": "c1",
            "check": "check 1",
            "scoring_rule": "rule 1",
            "max_points": 10,
        },
        {
            "id": "c2",
            "check": "check 2",
            "scoring_rule": "rule 2",
            "max_points": 5,
        },
    ]


def _valid_train_payload() -> dict:
    return {
        "criteria_scores": [
            {"id": "c1", "score": 7, "max": 10, "reason": "supported"},
            {"id": "c2", "score": 4, "max": 5, "reason": "mostly supported"},
        ],
        "total_score": 11,
        "total_max": 15,
    }


def test_val_v9_prompt_fills_placeholders_and_uses_runner_layout(monkeypatch):
    monkeypatch.setattr(
        rubric_scripts,
        "_val_judge_template",
        "specification\nQuery={QUERY}\nCandidate={RESPONSE}",
    )

    messages = rubric_scripts._build_val_judge_messages("brief", "script text")

    assert messages[0]["role"] == "system"
    assert "untrusted data" in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "specification\nQuery=brief\nCandidate=script text",
    }


def test_val_v9_parser_uses_canonical_score_and_candidate(monkeypatch):
    seen = {}

    def validate(payload, candidate_text):
        seen["payload"] = payload
        seen["candidate_text"] = candidate_text
        return 75.0

    monkeypatch.setattr(validate_overall_v9, "extract_validated_v9_score", validate)
    payload = {"rubric_version": "microfilm_overall_12_v9"}

    score, diagnostics = rubric_scripts._validate_val_judge_output(payload, "candidate script")

    assert score == 75.0
    assert seen == {"payload": payload, "candidate_text": "candidate script"}
    assert {key: diagnostics[key] for key in "ABCDEF"} == {key: 0.0 for key in "ABCDEF"}


def test_v9_validator_fails_closed_on_wrong_schema():
    with pytest.raises(ValueError, match="expected microfilm_overall_12_v9"):
        validate_overall_v9.extract_validated_v9_score({"rubric_version": "legacy"}, "candidate")


def test_train_judge_parser_accepts_exact_schema():
    score, total_max, breakdown, normalized = rubric_scripts._validate_judge_output(
        _valid_train_payload(), _train_criteria()
    )

    assert score == 11
    assert total_max == 15
    assert breakdown == [
        {"id": "c1", "score": 7.0, "max": 10.0, "reason": "supported"},
        {"id": "c2", "score": 4.0, "max": 5.0, "reason": "mostly supported"},
    ]
    assert normalized is False


@pytest.mark.parametrize("first_max,second_max", [(10.0, 5.0), ("10", "5.0")])
def test_train_judge_parser_normalizes_integral_max_representations(first_max, second_max):
    payload = _valid_train_payload()
    payload["criteria_scores"][0]["max"] = first_max
    payload["criteria_scores"][1]["max"] = second_max

    score, total_max, breakdown, _ = rubric_scripts._validate_judge_output(payload, _train_criteria())

    assert score == 11
    assert total_max == 15
    assert [item["max"] for item in breakdown] == [10.0, 5.0]


@pytest.mark.parametrize("derived_max", [True, 10.5, "10.5", "ten", None, 100])
def test_train_judge_parser_uses_rubric_max_instead_of_returned_max(derived_max):
    payload = _valid_train_payload()
    payload["criteria_scores"][0]["max"] = derived_max

    score, total_max, breakdown, normalized = rubric_scripts._validate_judge_output(payload, _train_criteria())

    assert score == 11
    assert total_max == 15
    assert breakdown[0]["max"] == 10.0
    assert normalized is True


def test_train_judge_parser_accepts_missing_derived_max():
    payload = _valid_train_payload()
    del payload["criteria_scores"][0]["max"]

    score, total_max, breakdown, normalized = rubric_scripts._validate_judge_output(payload, _train_criteria())

    assert score == 11
    assert total_max == 15
    assert breakdown[0]["max"] == 10.0
    assert normalized is True


def test_train_judge_parser_still_rejects_score_above_rubric_max():
    payload = _valid_train_payload()
    payload["criteria_scores"][0].update({"score": 11, "max": 100})

    with pytest.raises(ValueError, match=r"score 11 outside \[0, 10\]"):
        rubric_scripts._validate_judge_output(payload, _train_criteria())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"criteria_scores": []},
        {
            "criteria_scores": [
                {"id": "c1", "score": 7, "max": 10, "reason": "ok"},
                {"id": "c1", "score": 4, "max": 5, "reason": "duplicate"},
            ],
            "total_score": 11,
            "total_max": 15,
        },
        {
            "criteria_scores": [
                {"id": "c1", "score": 7.5, "max": 10, "reason": "fractional"},
                {"id": "c2", "score": 4, "max": 5, "reason": "ok"},
            ],
            "total_score": 11,
            "total_max": 15,
        },
        {
            "criteria_scores": [
                {"id": "c1", "score": 7, "max": 10},
                {"id": "c2", "score": 4, "max": 5, "reason": "ok"},
            ],
            "total_score": 11,
            "total_max": 15,
        },
    ],
)
def test_train_judge_parser_rejects_malformed_payloads(payload):
    with pytest.raises(ValueError):
        rubric_scripts._validate_judge_output(payload, _train_criteria())


@pytest.mark.parametrize(
    "declared_score,declared_max",
    [(12, 15), (11, 16), (None, None)],
)
def test_train_judge_parser_normalizes_derived_totals(declared_score, declared_max):
    payload = _valid_train_payload()
    payload["total_score"] = declared_score
    payload["total_max"] = declared_max

    score, total_max, breakdown, normalized = rubric_scripts._validate_judge_output(payload, _train_criteria())

    assert score == 11
    assert total_max == 15
    assert len(breakdown) == 2
    assert normalized is True


def test_reward_aggregation_caps_are_validated_and_applied_mechanically():
    criteria = _train_criteria()
    aggregation = rubric_scripts._validate_reward_aggregation(
        {
            "reward_aggregation": {
                "type": "normalized_sum_with_caps",
                "caps": [
                    {
                        "criterion_id": "c1",
                        "score_lte": 7,
                        "max_normalized_score": 0.6,
                    }
                ],
            }
        },
        criteria,
    )

    normalized, triggered = rubric_scripts._apply_reward_aggregation(
        11 / 15,
        [
            {"id": "c1", "score": 7.0, "max": 10.0},
            {"id": "c2", "score": 4.0, "max": 5.0},
        ],
        aggregation,
    )

    assert normalized == 0.6
    assert triggered == [
        {
            "criterion_id": "c1",
            "score_lte": 7,
            "max_normalized_score": 0.6,
            "binding": True,
        }
    ]


def test_reward_aggregation_is_backward_compatible_without_config():
    assert rubric_scripts._validate_reward_aggregation({}, _train_criteria()) is None
    assert rubric_scripts._apply_reward_aggregation(0.75, [], None) == (0.75, [])


def test_reward_aggregation_rejects_unknown_criterion():
    with pytest.raises(ValueError, match="unknown criterion"):
        rubric_scripts._validate_reward_aggregation(
            {
                "reward_aggregation": {
                    "type": "normalized_sum_with_caps",
                    "caps": [
                        {
                            "criterion_id": "missing",
                            "score_lte": 0,
                            "max_normalized_score": 0.5,
                        }
                    ],
                }
            },
            _train_criteria(),
        )


def test_extract_content_accepts_openai_output_text():
    assert rubric_scripts._extract_content({"output_text": "native response"}) == "native response"
    assert rubric_scripts._extract_content("direct response") == "direct response"


def test_train_judge_logs_failure_then_retries(monkeypatch, capsys):
    responses = iter(
        [
            "",
            json.dumps(
                {
                    "criteria_scores": [
                        {
                            "id": "c1",
                            "score": 7,
                            "max": 10,
                            "reason": "supported",
                        }
                    ],
                    "total_score": 7,
                    "total_max": 10,
                }
            ),
        ]
    )
    monkeypatch.setattr(rubric_scripts, "_judge_template", "{QUERY}\n{RUBRIC}\n{RESPONSE}")
    monkeypatch.setattr(rubric_scripts, "_API_MAX_RETRIES", 3)
    monkeypatch.setattr(rubric_scripts, "_API_RETRY_SLEEP_SECS", 0.0)
    monkeypatch.setattr(
        rubric_scripts,
        "_call_llm_sync",
        lambda _messages, _model, _max_tokens, _temperature: next(responses),
    )

    async def run():
        manager = object.__new__(rubric_scripts.RubricScriptRewardManager)
        manager.loop = _ImmediateLoop()
        manager._judge_executor = None
        criteria = [
            {
                "id": "c1",
                "check": "check",
                "scoring_rule": "rule",
                "max_points": 10,
            }
        ]
        return await manager._score_one(
            query="brief",
            rubric_obj={"criteria": criteria},
            response="script",
            criteria=criteria,
            total_max=10,
        )

    result = asyncio.run(run())

    assert result["ok"] is True
    assert result["attempts"] == 2
    assert result["total_score"] == 7
    assert "[RubricScript:train] attempt 1/3 failed: empty_content" in capsys.readouterr().out


def test_train_judge_does_not_retry_derived_total_mismatch(monkeypatch):
    malformed = _valid_train_payload()
    malformed["total_score"] = 12
    calls = 0

    def respond(*_args):
        nonlocal calls
        calls += 1
        return json.dumps(malformed)

    monkeypatch.setattr(rubric_scripts, "_judge_template", "{QUERY}\n{RUBRIC}\n{RESPONSE}")
    monkeypatch.setattr(rubric_scripts, "_API_MAX_RETRIES", 2)
    monkeypatch.setattr(rubric_scripts, "_API_RETRY_SLEEP_SECS", 0.0)
    monkeypatch.setattr(
        rubric_scripts,
        "_call_llm_sync",
        respond,
    )

    async def run():
        manager = object.__new__(rubric_scripts.RubricScriptRewardManager)
        manager.loop = _ImmediateLoop()
        manager._judge_executor = None
        criteria = _train_criteria()
        return await manager._score_one(
            query="brief",
            rubric_obj={"criteria": criteria},
            response="script",
            criteria=criteria,
            total_max=15,
        )

    result = asyncio.run(run())

    assert result["ok"] is True
    assert calls == 1
    assert result["attempts"] == 1
    assert result["total_score"] == 11
    assert result["derived_fields_normalized"] is True


def test_grpo_masks_judge_failures_and_unrankable_groups():
    response_mask = torch.ones((6, 3), dtype=torch.float32)

    masked, masked_index, stats = _prepare_grpo_judge_inputs(
        response_mask=response_mask,
        index=np.array(["a", "a", "a", "b", "b", "c"], dtype=object),
        judge_valid=np.array([1, 0, 1, 1, 0, 1], dtype=float),
    )

    assert masked.sum(dim=-1).tolist() == [3.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    assert masked_index[0] == masked_index[2] == "a"
    assert len(set(masked_index[[1, 3, 4, 5]].tolist())) == 4
    assert stats == {
        "judge_invalid_samples": 2.0,
        "judge_unrankable_valid_samples": 2.0,
    }
