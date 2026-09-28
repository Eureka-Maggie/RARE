import asyncio
import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl.experimental.reward_loop.reward_manager.rubric_scripts_task import (
    RubricScriptTaskRewardManager,
    _parse_switch_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = REPO_ROOT / "rubric" / "task_level"
SWITCH_PROMPT = REPO_ROOT / "rubric" / "switch" / "task_level_script_rubric_evolve_v1.md"
LAUNCHER_DIR = REPO_ROOT / "scripts"


class _Tokenizer:
    pass


def _config():
    return OmegaConf.create({"reward": {"num_workers": 1}, "actor_rollout_ref": {"rollout": {"n": 8}}})


def _manager(monkeypatch, tmp_path, *, dynamic=True):
    monkeypatch.setenv("RUBRIC_SCRIPT_TASK_ID", "2d_animation")
    monkeypatch.setenv("RUBRIC_SCRIPT_TASK_DESCRIPTION", "2D animation scripts")
    monkeypatch.setenv("RUBRIC_SCRIPT_TASK_RUBRIC_PATH", str(TASK_DIR / "2d_animation_r0.json"))
    monkeypatch.setenv("RUBRIC_SCRIPT_TASK_SWITCHING_ENABLED", "1" if dynamic else "0")
    monkeypatch.setenv("RUBRIC_SCRIPT_TASK_SWITCH_STEPS", "22,44,66,88,110")
    monkeypatch.setenv("RUBRIC_SCRIPT_TASK_SWITCH_PROMPT_PATH", str(SWITCH_PROMPT))
    monkeypatch.setenv("RUBRIC_SCRIPT_TASK_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("RUBRIC_SCRIPT_TASK_LOG_PATH", str(tmp_path / "events.jsonl"))
    manager = RubricScriptTaskRewardManager(_config(), _Tokenizer(), None)
    return manager


def _close(manager):
    manager._judge_executor.shutdown(wait=False, cancel_futures=True)


def test_task_r0_rubrics_have_comparable_shape_and_distinct_medium_axis():
    rubrics = [json.loads(path.read_text("utf-8")) for path in sorted(TASK_DIR.glob("*_r0.json"))]
    assert len(rubrics) == 3
    expected_ids = [item["id"] for item in rubrics[0]["criteria"]]
    expected_maxima = [item["max_points"] for item in rubrics[0]["criteria"]]
    assert expected_maxima == [2, 2, 2, 2, 2, 2]
    assert sum(expected_maxima) == 12
    for rubric in rubrics:
        assert rubric["rubric_version"] == 0
        assert [item["id"] for item in rubric["criteria"]] == expected_ids
        assert [item["max_points"] for item in rubric["criteria"]] == expected_maxima
    medium_checks = {
        next(item["check"] for item in rubric["criteria"] if item["id"] == "c_medium_specific_visual_function")
        for rubric in rubrics
    }
    assert len(medium_checks) == 3


def test_switch_step_parser_is_fail_closed():
    assert _parse_switch_steps("16,31,47,63,78") == (16, 31, 47, 63, 78)
    with pytest.raises(ValueError):
        _parse_switch_steps("22,22,44")
    with pytest.raises(ValueError):
        _parse_switch_steps("44,22")


@pytest.mark.parametrize(
    ("task", "switch_steps", "total_steps", "save_freq"),
    [
        ("2d_animation", "22,44,66,88,110", 132, 22),
        ("3d_stopmotion", "18,36,54,72,90", 108, 18),
        ("live_action", "16,31,47,63,78", 94, 16),
    ],
)
def test_task_schedules_are_preserved_in_the_shared_launcher(task, switch_steps, total_steps, save_freq):
    runner = (LAUNCHER_DIR / "train.py").read_text("utf-8")
    assert f'"{task}": {{' in runner
    assert f'"switch_steps": "{switch_steps}"' in runner
    assert f'"save_freq": {save_freq}' in runner
    assert f'"test_freq": {total_steps}' in runner


def test_task_entrypoint_preserves_train_val_judge_split_and_sample_metrics():
    runner = (LAUNCHER_DIR / "train.py").read_text("utf-8")
    assert "RUBRIC_SCRIPT_TASK_SWITCHING_ENABLED" in runner
    assert "RUBRIC_SCRIPT_SWITCHING_ENABLED" in runner
    assert "rubric/judge/judge_scripts_v1.md" in runner
    assert "rubric/judge/overall_judge_v9.txt" in runner
    assert "rubric_should_switch_count" in runner
    assert "rubric_switch_rescored_current" in runner
    assert "task_rubric_switch_committed" in runner


def test_evidence_selection_is_unique_bounded_and_spans_score_range(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)
    try:
        for step in (20, 21, 22):
            rows = []
            for index in range(32):
                score = index / 31
                query = f"query-{step}-{index}"
                rows.append(
                    {
                        "uid": f"uid-{step}-{index}",
                        "query": query,
                        "query_sha256": f"{step:02d}{index:02d}",
                        "group_mean": score,
                        "group_valid_count": 8,
                        "rubric_version": 0,
                        "representative": {
                            "rollout_index": 0,
                            "response": f"script-{step}-{index}",
                            "judge_valid": True,
                            "normalized_score": score,
                            "breakdown": [],
                        },
                    }
                )
            manager._task_recent_steps[step] = rows
        selected = manager._select_evidence(22)
        assert len(selected) == 16
        assert len({item["query_sha256"] for item in selected}) == 16
        scores = [item["group_mean"] for item in selected]
        assert min(scores) < 0.34
        assert max(scores) > 0.66
    finally:
        _close(manager)


def test_candidate_validator_preserves_ids_weights_and_next_version(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)
    try:
        candidate = copy.deepcopy(manager._task_active_rubric)
        candidate.update(
            {
                "rubric_version": 1,
                "generated_at_step": 22,
                "generated_from_trigger": "scheduled_recent_rollouts",
            }
        )
        candidate["criteria"][0]["check"] += "（新版边界）"
        clean = manager._validate_task_candidate(candidate, step=22)
        assert clean["rubric_version"] == 1
        assert sum(item["max_points"] for item in clean["criteria"]) == 12

        bad = copy.deepcopy(candidate)
        bad["criteria"][0]["max_points"] = 3
        with pytest.raises(ValueError, match="max_points"):
            manager._validate_task_candidate(bad, step=22)
    finally:
        _close(manager)


def test_all_rollouts_wait_for_one_complete_step_outcome(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)

    async def exercise():
        groups = [
            await manager._resolve_task_assignment(
                uid=f"uid-{group_index}",
                query=f"query-{group_index}",
                global_step=1,
                expected_step_groups=2,
            )
            for group_index in range(2)
        ]
        tasks = []
        for group in groups:
            for rollout_index in range(8):
                tasks.append(
                    asyncio.create_task(
                        manager._observe_task_assignment(
                            group=group,
                            rollout_index=rollout_index,
                            response=f"script-{group.uid}-{rollout_index}",
                            judge_valid=True,
                            normalized_score=rollout_index / 7,
                            breakdown=[{"id": "c_brief_grounding", "score": 1, "max": 2, "reason": "evidence"}],
                        )
                    )
                )
        outcomes = await asyncio.gather(*tasks)
        await asyncio.sleep(0)
        return outcomes

    try:
        outcomes = manager.loop.run_until_complete(exercise())
        assert len(outcomes) == 16
        assert all(not outcome["due"] and not outcome["attempted"] for outcome in outcomes)
        assert len(manager._task_recent_steps[1]) == 2
        assert 1 not in manager._task_steps
        assert manager._task_state_path.is_file()
    finally:
        _close(manager)


def test_committed_switch_is_used_only_by_the_next_step(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)
    for step in (20, 21, 22):
        manager._task_recent_steps[step] = [
            {
                "uid": f"uid-{step}-{index}",
                "query": f"query-{step}-{index}",
                "query_sha256": f"hash-{step}-{index}",
                "group_mean": index / 31,
                "group_valid_count": 8,
                "rubric_version": 0,
                "representative": {
                    "rollout_index": 3,
                    "response": f"script-{step}-{index}",
                    "judge_valid": True,
                    "normalized_score": index / 31,
                    "breakdown": [],
                },
            }
            for index in range(32)
        ]

    candidate = copy.deepcopy(manager._task_active_rubric)
    candidate.update(
        {
            "rubric_version": 1,
            "generated_at_step": 22,
            "generated_from_trigger": "scheduled_recent_rollouts",
        }
    )
    candidate["criteria"][0]["check"] += "（R1）"

    async def fake_generate(*, step, evidence):
        assert step == 22 and 12 <= len(evidence) <= 16
        return candidate, 1, ""

    async def fake_shadow(candidate_arg, evidence):
        assert candidate_arg["rubric_version"] == 1
        return True, {
            "old_mean": 0.5,
            "new_mean": 0.45,
            "mean_shift": 0.05,
            "new_unique_scores": 3,
            "accepted": True,
            "reason": "passed",
        }

    monkeypatch.setattr(manager, "_generate_task_candidate", fake_generate)
    monkeypatch.setattr(manager, "_shadow_gate", fake_shadow)

    async def exercise():
        # A group resolved before the step-22 commit keeps its immutable R0.
        current = await manager._resolve_task_assignment(
            uid="current-step", query="current", global_step=22, expected_step_groups=1
        )
        outcome = await manager._run_scheduled_switch(22)
        following = await manager._resolve_task_assignment(
            uid="next-step", query="next", global_step=23, expected_step_groups=1
        )
        return current, outcome, following

    try:
        current, outcome, following = manager.loop.run_until_complete(exercise())
        assert current.rubric_version == 0
        assert outcome["committed"]
        assert following.rubric_version == 1
        assert manager._task_active_version == 1
        assert manager._task_applied_steps == {22}
    finally:
        _close(manager)


def test_resume_rolls_back_rubric_state_ahead_of_model_checkpoint(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)
    try:
        r1 = copy.deepcopy(manager._task_active_rubric)
        r1["rubric_version"] = 1
        r1["generated_at_step"] = 22
        r1["criteria"][0]["check"] += "（R1）"
        manager._task_history.append({"version": 1, "source_step": 22, "rubric": r1})
        manager._task_active_rubric = r1
        manager._task_active_version = 1
        manager._task_updated_step = 22
        manager._task_applied_steps = {22}
        manager._save_task_state()
    finally:
        _close(manager)

    resumed = _manager(monkeypatch, tmp_path)
    try:
        assert resumed._task_active_version == 1
        resumed._reconcile_resume_step(21)
        assert resumed._task_active_version == 0
        assert resumed._task_applied_steps == set()
        assert len(resumed._task_history) == 1
    finally:
        _close(resumed)
