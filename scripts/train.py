#!/usr/bin/env python3
"""Validated launcher for the six released rubric-training variants."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "scripts" / "video_corpus_manual_v1_1b_20260807"

METHODS = (
    "sample-static",
    "sample-dynamic-negative",
    "sample-dynamic-positive",
    "sample-dynamic-mix",
    "task-static",
    "task-dynamic",
)

TASKS = {
    "2d_animation": {
        "description": (
            "Generate short screenplays for 2D animation; visual design, motion, transitions, "
            "and color should serve the narrative."
        ),
        "rubric": "rubric/task_level/2d_animation_r0.json",
        "switch_steps": "22,44,66,88,110",
        "save_freq": 22,
        "test_freq": 132,
    },
    "3d_stopmotion": {
        "description": (
            "Generate short screenplays for 3D animation or stop motion; materials, object motion, "
            "scale, and volumetric lighting should serve the narrative."
        ),
        "rubric": "rubric/task_level/3d_stopmotion_r0.json",
        "switch_steps": "18,36,54,72,90",
        "save_freq": 18,
        "test_freq": 108,
    },
    "live_action": {
        "description": (
            "Generate live-action short screenplays; performance, blocking, props, locations, "
            "and continuous action should serve the narrative."
        ),
        "rubric": "rubric/task_level/live_action_r0.json",
        "switch_steps": "16,31,47,63,78",
        "save_freq": 16,
        "test_freq": 94,
    },
}

BASE_EXTRA_LOG_KEYS = [
    "rubric_score",
    "rubric_raw_score",
    "rubric_cap_applied",
    "rubric_total_score",
    "rubric_total_max",
    "rubric_num_criteria",
    "fallback_used",
    "judge_attempts",
    "judge_derived_fields_normalized",
    "judge_valid",
    "is_val",
    "rubric_should_switch_count",
    "rubric_saturated_criteria_per_sample",
    "rubric_saturated_sample",
    "rubric_switch_rescored_current",
]

TASK_EXTRA_LOG_KEYS = [
    "task_rubric_version",
    "task_rubric_switch_due",
    "task_rubric_switch_attempted",
    "task_rubric_switch_committed",
    "task_rubric_switch_skipped",
    "task_rubric_switch_timeout",
    "task_rubric_evidence_count",
    "task_rubric_writer_attempts",
    "task_rubric_shadow_old_mean",
    "task_rubric_shadow_new_mean",
]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), default=os.getenv("TASK", "3d_stopmotion"))
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH"))
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("OUTPUT_DIR", ROOT / "checkpoints")))
    parser.add_argument("--experiment-name", default=os.getenv("EXPERIMENT_NAME"))
    parser.add_argument("--seed", type=int, default=_env_int("TRAINING_SEED", 1))
    parser.add_argument("--batch-size", type=int, default=_env_int("TRAIN_BATCH_SIZE", 32))
    parser.add_argument("--epochs", type=int, default=_env_int("TOTAL_EPOCHS", 2))
    parser.add_argument("--gpus-per-node", type=int, default=_env_int("N_GPUS", 8))
    parser.add_argument("--nodes", type=int, default=_env_int("N_NODES", 1))
    parser.add_argument("--tensor-parallel-size", type=int, default=_env_int("TENSOR_PARALLEL_SIZE", 2))
    parser.add_argument("--judge-concurrency", type=int, default=_env_int("JUDGE_CONCURRENCY", 8))
    parser.add_argument("--validation-concurrency", type=int, default=_env_int("VAL_JUDGE_CONCURRENCY", 4))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def _require_file(path: Path, label: str) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.is_file():
        raise SystemExit(f"{label} not found: {resolved}")
    return resolved.resolve()


def configure_environment(args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "RUBRIC_SCRIPT_JUDGE_PROMPT_PATH": str(ROOT / "rubric/judge/judge_scripts_v1.md"),
            "RUBRIC_SCRIPT_VAL_JUDGE_PROMPT_PATH": str(ROOT / "rubric/judge/overall_judge_v9.txt"),
            "RUBRIC_SCRIPT_JUDGE_CONCURRENCY": str(args.judge_concurrency),
            "RUBRIC_SCRIPT_VAL_JUDGE_CONCURRENCY": str(args.validation_concurrency),
            "RUBRIC_SCRIPT_SWITCH_STATE_PATH": str(run_dir / "rubric_switch_state.json"),
            "RUBRIC_SCRIPT_SWITCH_LOG_PATH": str(run_dir / "rubric_switch_events.jsonl"),
            "RUBRIC_SCRIPT_SWITCH_STRATEGY": "joint",
            "RUBRIC_SCRIPT_SWITCH_PROMPT_PATH": str(ROOT / "rubric/switch/rubric_switch_joint_v1.md"),
            "RUBRIC_SCRIPT_SWITCH_RESCORE_PROMPT_PATH": str(ROOT / "rubric/judge/judge_joint_switch_v1.md"),
            "RUBRIC_SCRIPT_SWITCH_RESCORE_CURRENT_STEP": "1",
            "RUBRIC_SCRIPT_SWITCH_MAX_UPDATES": os.getenv("RUBRIC_SCRIPT_SWITCH_MAX_UPDATES", "3"),
        }
    )

    if args.method.startswith("sample-dynamic-"):
        env["RUBRIC_SCRIPT_SWITCHING_ENABLED"] = "1"
        env["RUBRIC_SCRIPT_SWITCH_POLARITY"] = args.method.rsplit("-", 1)[-1]
    else:
        env["RUBRIC_SCRIPT_SWITCHING_ENABLED"] = "0"
        env["RUBRIC_SCRIPT_SWITCH_POLARITY"] = "negative"

    if args.method.startswith("task-"):
        task = TASKS[args.task]
        env.update(
            {
                "RUBRIC_SCRIPT_TASK_ID": args.task,
                "RUBRIC_SCRIPT_TASK_DESCRIPTION": task["description"],
                "RUBRIC_SCRIPT_TASK_RUBRIC_PATH": str(ROOT / task["rubric"]),
                "RUBRIC_SCRIPT_TASK_SWITCHING_ENABLED": "1" if args.method == "task-dynamic" else "0",
                "RUBRIC_SCRIPT_TASK_SWITCH_STEPS": task["switch_steps"],
                "RUBRIC_SCRIPT_TASK_SWITCH_PROMPT_PATH": str(
                    ROOT / "rubric/switch/task_level_script_rubric_evolve_v1.md"
                ),
                "RUBRIC_SCRIPT_TASK_TRAINING_SEED": str(args.seed),
                "RUBRIC_SCRIPT_TASK_EVIDENCE_WINDOW_STEPS": "3",
                "RUBRIC_SCRIPT_TASK_MAX_EVIDENCE_GROUPS": "16",
                "RUBRIC_SCRIPT_TASK_MIN_EVIDENCE_GROUPS": "12",
                "RUBRIC_SCRIPT_TASK_STATE_PATH": str(run_dir / "task_rubric_state.json"),
                "RUBRIC_SCRIPT_TASK_LOG_PATH": str(run_dir / "task_rubric_events.jsonl"),
            }
        )
    return env


def build_command(args: argparse.Namespace, extra: list[str]) -> tuple[list[str], dict[str, str]]:
    if not args.model_path:
        raise SystemExit("--model-path or MODEL_PATH is required")
    model_path = Path(args.model_path).expanduser().resolve()
    _require_file(model_path / "config.json", "model config")

    train_file = _require_file(DATA_ROOT / args.task / "train.parquet", "training data")
    val_file = _require_file(DATA_ROOT / args.task / "test.parquet", "validation data")
    for path in (
        ROOT / "rubric/judge/judge_scripts_v1.md",
        ROOT / "rubric/judge/overall_judge_v9.txt",
        ROOT / "rubric/judge/overall_v9_validator.py",
        ROOT / "rubric/judge/judge_joint_switch_v1.md",
        ROOT / "rubric/switch/rubric_switch_joint_v1.md",
    ):
        _require_file(path, "rubric asset")

    if args.method.startswith("task-"):
        _require_file(ROOT / TASKS[args.task]["rubric"], "task rubric")
        _require_file(ROOT / "rubric/switch/task_level_script_rubric_evolve_v1.md", "task switch prompt")

    method_slug = args.method.replace("-", "_")
    experiment = args.experiment_name or f"{method_slug}_{args.task}_seed{args.seed}_bs{args.batch_size}"
    run_dir = args.output_dir.expanduser().resolve() / experiment
    task = TASKS[args.task]
    reward_manager = "rubric_script_task" if args.method.startswith("task-") else "rubric_script"
    extra_keys = BASE_EXTRA_LOG_KEYS + (TASK_EXTRA_LOG_KEYS if args.method.startswith("task-") else [])

    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_file}",
        f"data.val_files=[{val_file}]",
        f"data.seed={args.seed}",
        f"data.train_batch_size={args.batch_size}",
        "data.max_prompt_length=4096",
        "data.max_response_length=8192",
        "data.filter_overlong_prompts=true",
        "data.truncation=left",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={_env_int('PPO_MICRO_BATCH_SIZE_PER_GPU', 4)}",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tensor_parallel_size}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={os.getenv('GPU_MEMORY_UTILIZATION', '0.7')}",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32",
        "actor_rollout_ref.ref.fsdp_config.param_offload=false",
        "algorithm.use_kl_in_reward=false",
        f"+algorithm.reward_extra_log_keys=[{','.join(extra_keys)}]",
        f"reward.reward_manager.name={reward_manager}",
        "reward.num_workers=1",
        f"+reward.rubric_switching_enabled={'true' if args.method.startswith('sample-dynamic-') else 'false'}",
        "+reward.rubric_switch_rescore_current_step=true",
        f"+reward.rubric_switch_apply_mode={method_slug}",
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=evolving_rubrics",
        f"trainer.experiment_name={experiment}",
        f"trainer.n_gpus_per_node={args.gpus_per_node}",
        f"trainer.nnodes={args.nodes}",
        f"trainer.save_freq={task['save_freq']}",
        f"trainer.test_freq={task['test_freq']}",
        "trainer.val_before_train=false",
        "trainer.total_training_steps=null",
        f"trainer.total_epochs={args.epochs}",
        "trainer.resume_mode=auto",
        "actor_rollout_ref.rollout.val_kwargs.n=4",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        f"trainer.validation_data_dir={run_dir / 'validation_generations'}",
        f"trainer.default_local_dir={run_dir}",
        *extra,
    ]
    return command, configure_environment(args, run_dir)


def main() -> int:
    args, extra = parse_args()
    command, env = build_command(args, extra)
    if args.dry_run:
        print(json.dumps({"command": command, "cwd": str(ROOT)}, indent=2))
        return 0
    if not env.get("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is required")
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
