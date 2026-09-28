import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_train", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train)


@pytest.mark.parametrize("method", train.METHODS)
def test_every_released_method_builds_a_complete_command(tmp_path, method):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", "utf-8")
    args = argparse.Namespace(
        method=method,
        task="3d_stopmotion",
        model_path=str(model),
        output_dir=tmp_path / "outputs",
        experiment_name=None,
        seed=1,
        batch_size=32,
        epochs=2,
        gpus_per_node=8,
        nodes=1,
        tensor_parallel_size=2,
        judge_concurrency=8,
        validation_concurrency=4,
        dry_run=True,
    )

    command, env = train.build_command(args, [])
    joined = "\n".join(command)

    assert "verl.trainer.main_ppo" in command
    assert "actor_rollout_ref.rollout.n=8" in command
    assert "reward.num_workers=1" in command
    assert 'trainer.logger=["console"]' in command
    assert "3d_stopmotion/train.parquet" in joined
    assert "3d_stopmotion/test.parquet" in joined
    assert "RUBRIC_SCRIPT_SWITCH_STRATEGY" in env

    if method.startswith("sample-dynamic-"):
        assert env["RUBRIC_SCRIPT_SWITCHING_ENABLED"] == "1"
        assert env["RUBRIC_SCRIPT_SWITCH_POLARITY"] == method.rsplit("-", 1)[-1]
        assert "reward.reward_manager.name=rubric_script" in command
    elif method == "sample-static":
        assert env["RUBRIC_SCRIPT_SWITCHING_ENABLED"] == "0"
        assert "reward.reward_manager.name=rubric_script" in command
    else:
        assert env["RUBRIC_SCRIPT_SWITCHING_ENABLED"] == "0"
        assert "reward.reward_manager.name=rubric_script_task" in command
        assert env["RUBRIC_SCRIPT_TASK_SWITCHING_ENABLED"] == ("1" if method == "task-dynamic" else "0")


def test_six_shell_launchers_are_the_only_shell_files():
    shell_files = sorted(path.name for path in (ROOT / "scripts").glob("*.sh"))
    assert shell_files == [
        "train_sample_dynamic_mix.sh",
        "train_sample_dynamic_negative.sh",
        "train_sample_dynamic_positive.sh",
        "train_sample_static.sh",
        "train_task_dynamic.sh",
        "train_task_static.sh",
    ]
