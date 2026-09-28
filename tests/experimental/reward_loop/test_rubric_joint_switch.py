import asyncio
import copy
import json

import pytest

from verl.experimental.reward_loop.reward_manager import rubric_scripts
from verl.experimental.reward_loop.reward_manager.rubric_script_switch import (
    render_joint_switch_prompt,
    validate_joint_score_output,
    validate_joint_switch_output,
)


def _rubric():
    return {
        "rubric_version": 1,
        "criteria": [
            {
                "id": f"c{index}",
                "check": f"old check {index}",
                "scoring_rule": "2 = strong; 1 = partial; 0 = absent",
                "max_points": 2,
            }
            for index in range(1, 5)
        ],
    }


def _observations():
    return [
        {
            "response": f"rollout {index}",
            "judge_valid": True,
            "scores": {f"c{criterion}": 2 for criterion in range(1, 5)},
            "reasons": {f"c{criterion}": f"reason {index}-{criterion}" for criterion in range(1, 5)},
            "breakdown": [
                {
                    "id": f"c{criterion}",
                    "score": 2.0,
                    "max": 2.0,
                    "reason": f"reason {index}-{criterion}",
                }
                for criterion in range(1, 5)
            ],
        }
        for index in range(8)
    ]


def _writer_payload(ids=("c1", "c2", "c3"), polarities=None):
    polarities = polarities or ("negative",) * len(ids)
    return {
        "updates": [
            {
                "id": criterion_id,
                "polarity": polarity,
                "diagnosis": {
                    "difference": f"difference for {criterion_id}",
                    "best_rollout_ids": ["R0", "R1"],
                    "weaker_rollout_ids": ["R6", "R7"],
                    "learning_direction": f"direction for {criterion_id}",
                    "nonredundancy": f"distinct evidence for {criterion_id}",
                },
                "rubric": {
                    "id": criterion_id,
                    "check": f"new check {criterion_id}",
                    "scoring_rule": "2 = clear; 1 = partial; 0 = harmful",
                    "max_points": 2,
                },
            }
            for criterion_id, polarity in zip(ids, polarities, strict=True)
        ]
    }


def _score_payload(ids=("c1", "c2", "c3")):
    return {
        "criteria": [
            {
                "id": criterion_id,
                "scores": [
                    {
                        "rollout_id": f"R{index}",
                        "score": index % 3,
                        "reason": f"evidence {criterion_id}-{index}",
                    }
                    for index in range(8)
                ],
            }
            for criterion_id in ids
        ]
    }


@pytest.mark.parametrize(
    "polarity,polarities",
    [
        ("negative", ("negative", "negative", "negative")),
        ("positive", ("positive", "positive", "positive")),
        ("mix", ("negative", "positive", "negative")),
    ],
)
def test_joint_writer_contract_supports_all_polarities(polarity, polarities):
    result = validate_joint_switch_output(
        _writer_payload(polarities=polarities),
        current_rubric=_rubric(),
        saturated_ids=["c1", "c2", "c3", "c4"],
        response_count=8,
        update_count=3,
        polarity=polarity,
    )

    assert [row["polarity"] for row in result["updates"]] == list(polarities)


def test_joint_writer_rejects_wrong_count_order_and_one_sided_mix():
    with pytest.raises(ValueError, match="update count"):
        validate_joint_switch_output(
            _writer_payload(ids=("c1", "c2")),
            current_rubric=_rubric(),
            saturated_ids=["c1", "c2", "c3", "c4"],
            response_count=8,
            update_count=3,
            polarity="negative",
        )
    with pytest.raises(ValueError, match="source-ordered"):
        validate_joint_switch_output(
            _writer_payload(ids=("c2", "c1", "c3")),
            current_rubric=_rubric(),
            saturated_ids=["c1", "c2", "c3", "c4"],
            response_count=8,
            update_count=3,
            polarity="negative",
        )
    with pytest.raises(ValueError, match="mix requires"):
        validate_joint_switch_output(
            _writer_payload(),
            current_rubric=_rubric(),
            saturated_ids=["c1", "c2", "c3", "c4"],
            response_count=8,
            update_count=3,
            polarity="mix",
        )


def test_joint_prompt_contains_all_saturated_items_but_requests_cap():
    prompt = render_joint_switch_prompt(
        "count={UPDATE_COUNT}\npolarity={POLARITY_INSTRUCTION}\n"
        "sat={SATURATED_CRITERIA}\nrollouts={CURRENT_ROLLOUTS_WITH_SCORES}",
        query="brief",
        current_rubric=_rubric(),
        saturated_ids=["c1", "c2", "c3", "c4"],
        observations=_observations(),
        global_step=7,
        update_count=3,
        polarity="negative",
    )

    assert "count=3" in prompt
    assert all(f'"id": "c{index}"' in prompt for index in range(1, 5))
    assert "### R0" in prompt and "### R7" in prompt


@pytest.mark.parametrize("mutation", ["missing", "reordered", "duplicate", "bool", "range"])
def test_joint_score_matrix_is_strict_and_fail_closed(mutation):
    payload = _score_payload(("c1", "c2"))
    if mutation == "missing":
        payload["criteria"][0]["scores"].pop()
    elif mutation == "reordered":
        payload["criteria"].reverse()
    elif mutation == "duplicate":
        payload["criteria"][0]["scores"][1]["rollout_id"] = "R0"
    elif mutation == "bool":
        payload["criteria"][0]["scores"][0]["score"] = True
    elif mutation == "range":
        payload["criteria"][0]["scores"][0]["score"] = 3

    with pytest.raises(ValueError):
        validate_joint_score_output(
            payload,
            criteria=_rubric()["criteria"][:2],
            response_count=8,
        )


def _manager():
    manager = object.__new__(rubric_scripts.RubricScriptRewardManager)
    manager.loop = asyncio.get_running_loop()
    manager._switch_lock = None
    manager._switch_sem = None
    manager._judge_sem = None
    manager._judge_concurrency = 12
    manager._switch_executor = None
    manager._judge_executor = None
    manager._active_rubrics = {}
    manager._active_versions = {}
    manager._switch_inflight = {"stem"}
    manager._switch_template = "template"
    manager._joint_rescore_template = "{QUERY}\n{RUBRICS}\n{ROLLOUTS}"
    return manager


def _group(manager):
    return rubric_scripts._SwitchGroup(
        uid="uid",
        stem_uid="stem",
        query="brief",
        global_step=5,
        expected_n=8,
        expected_step_groups=1,
        rubric=_rubric(),
        rubric_version=0,
        observations=_observations(),
        decision_future=manager.loop.create_future(),
    )


def test_joint_online_path_uses_one_writer_and_one_judge_call(monkeypatch):
    async def run():
        manager = _manager()
        group = _group(manager)
        calls = {"writer": 0, "judge": 0}

        def fake_writer(_messages):
            calls["writer"] += 1
            return json.dumps(_writer_payload())

        def fake_judge(_messages, _model, _max_tokens, _temperature):
            calls["judge"] += 1
            return json.dumps(_score_payload())

        monkeypatch.setattr(rubric_scripts, "_call_switch_llm_sync", fake_writer)
        monkeypatch.setattr(rubric_scripts, "_call_llm_sync", fake_judge)
        monkeypatch.setattr(rubric_scripts, "_SWITCH_POLARITY", "negative")
        monkeypatch.setattr(rubric_scripts, "_SWITCH_MAX_UPDATES", 3)
        monkeypatch.setattr(rubric_scripts, "_SWITCH_API_MAX_RETRIES", 1)
        monkeypatch.setattr(rubric_scripts, "_API_MAX_RETRIES", 1)

        await manager._run_joint_current_step_switch(group, ["c1", "c2", "c3", "c4"])
        return manager, group, calls

    manager, group, calls = asyncio.run(run())
    outcome = group.decision_future.result()
    assert calls == {"writer": 1, "judge": 1}
    assert outcome["target_ids"] == ["c1", "c2", "c3"]
    assert outcome["candidate_rubric"]["criteria"][3]["check"] == "old check 4"
    assert len(outcome["rescores"]) == 8
    assert manager._active_versions == {"stem": 1}
    assert manager._active_rubrics["stem"]["rubric_version"] == 2


@pytest.mark.parametrize("failed_stage", ["writer", "judge"])
def test_joint_online_failure_preserves_old_reward_and_state(monkeypatch, failed_stage):
    async def run():
        manager = _manager()
        group = _group(manager)

        def fake_writer(_messages):
            if failed_stage == "writer":
                raise RuntimeError("writer unavailable")
            return json.dumps(_writer_payload())

        def fake_judge(_messages, _model, _max_tokens, _temperature):
            if failed_stage == "judge":
                raise RuntimeError("judge unavailable")
            return json.dumps(_score_payload())

        monkeypatch.setattr(rubric_scripts, "_call_switch_llm_sync", fake_writer)
        monkeypatch.setattr(rubric_scripts, "_call_llm_sync", fake_judge)
        monkeypatch.setattr(rubric_scripts, "_SWITCH_POLARITY", "negative")
        monkeypatch.setattr(rubric_scripts, "_SWITCH_MAX_UPDATES", 3)
        monkeypatch.setattr(rubric_scripts, "_SWITCH_API_MAX_RETRIES", 1)
        monkeypatch.setattr(rubric_scripts, "_API_MAX_RETRIES", 1)

        old_rubric = copy.deepcopy(group.rubric)
        await manager._run_joint_current_step_switch(group, ["c1", "c2", "c3"])
        return manager, group, old_rubric

    manager, group, old_rubric = asyncio.run(run())
    assert group.decision_future.result() == {
        "target_ids": [],
        "candidate_rubric": None,
        "rescores": None,
    }
    assert group.rubric == old_rubric
    assert manager._active_rubrics == {}
    assert manager._active_versions == {}
