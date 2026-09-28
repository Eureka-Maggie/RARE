# Paper-to-code guide

This guide maps the released RARE experiment families to their launchers and
runtime implementation. It covers the six variants supported by this public
release.

## Experiment matrix

| Paper setting | Launcher | Reward manager | Rubric updates |
| --- | --- | --- | --- |
| Sample-Static | [`train_sample_static.sh`](../scripts/train_sample_static.sh) | `rubric_script` | Disabled |
| Sample-Dynamic (Negative) | [`train_sample_dynamic_negative.sh`](../scripts/train_sample_dynamic_negative.sh) | `rubric_script` | Per query, negative criteria |
| Sample-Dynamic (Positive) | [`train_sample_dynamic_positive.sh`](../scripts/train_sample_dynamic_positive.sh) | `rubric_script` | Per query, positive criteria |
| Sample-Dynamic (Mixed) | [`train_sample_dynamic_mix.sh`](../scripts/train_sample_dynamic_mix.sh) | `rubric_script` | Per query, both directions |
| Task-Static | [`train_task_static.sh`](../scripts/train_task_static.sh) | `rubric_script_task` | Disabled |
| Task-Dynamic | [`train_task_dynamic.sh`](../scripts/train_task_dynamic.sh) | `rubric_script_task` | Scheduled, task-wide |

All six wrappers call [`train.py`](../scripts/train.py), which validates the
model checkpoint, annotation files, and rubric assets before constructing the
Hydra command. Use `--dry-run` to inspect that command without starting Ray or
calling an external model.

## Sample-level evolution

The sample-level implementation is in
[`rubric_scripts.py`](../verl/experimental/reward_loop/reward_manager/rubric_scripts.py),
with replacement validation helpers in
[`rubric_script_switch.py`](../verl/experimental/reward_loop/reward_manager/rubric_script_switch.py).

For each query group:

1. The judge scores all eight current-policy rollouts against the active rubric.
2. A criterion is saturated only when all eight valid judgments receive its
   maximum score.
3. The rubric writer replaces up to three saturated criteria while preserving
   criterion IDs, maximum scores, and all untouched criteria.
4. The replacement criteria are rescored on the same eight outputs.
5. The current reward combines replacement scores with retained original
   scores, so a committed update affects the current policy update.

Negative replacements describe observable defects while keeping larger scores
better. Positive replacements describe desired qualities. Mixed replacements
may use either direction and, when at least two criteria are replaced, must
contain both directions.

Failed judge calls use zero only as a tensor placeholder. The accompanying
`judge_valid` signal excludes those observations from saturation and group
comparisons.

## Task-level evolution

Task-level behavior is implemented in
[`rubric_scripts_task.py`](../verl/experimental/reward_loop/reward_manager/rubric_scripts_task.py).
Each production style begins with its corresponding rubric in
[`rubric/task_level`](../rubric/task_level). The dynamic launcher updates this
shared task rubric at the schedule defined in [`train.py`](../scripts/train.py),
using valid evidence collected from recent rollout groups. State and event logs
are written inside the experiment output directory, allowing an interrupted run
to resume without silently repeating a committed update.

The static task-level launcher uses the same initial rubric and reward manager
with scheduled rewriting disabled.

## Data and reproducibility boundary

The public [Hugging Face dataset](https://huggingface.co/datasets/EurekaTian/RARE)
contains the exact query-rubric schema consumed by the launchers. Source-video
titles and uploader handles are replaced by anonymous stable IDs, and original
videos are not redistributed. The checked-in
[`release_manifest.json`](../release_manifest.json) records row counts and
SHA-256 digests for all six annotation files.

The release intentionally excludes private infrastructure configuration,
experiment-tracking credentials, model checkpoints, and generated training
outputs.
