# RARE environment

RARE uses the `verl` snapshot included in this repository. A separate `verl`
checkout is not required, and replacing the bundled snapshot with a newer
release may change rollout, masking, or checkpoint behavior.

## Reference stack

The released training recipe was validated with the following stack:

| Component | Reference value |
| --- | --- |
| Platform | Linux x86_64 |
| Python | 3.12.13 |
| GPUs | 8 × NVIDIA L20Z 80 GB |
| NVIDIA driver | 580.105.08 |
| Host-reported CUDA | 12.8 |
| PyTorch | 2.7.0+cu126 |
| PyTorch CUDA runtime | 12.6 |
| cuDNN | 9.5.0.1 |
| vLLM | 0.9.2 |
| Transformers | 4.51.1 |
| Ray | 2.43.0 |
| FlashAttention | 2.7.4.post1 |
| FlashInfer | 0.3.1 |

Eight 80 GB GPUs are the reference hardware for the default launchers. Other
GPU types can work, but batch size, rollout memory, and compiled CUDA packages
may need adjustment.

## Recreate the environment

Create a Python 3.12 environment with `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install setuptools==79.0.1 wheel==0.46.3 \
  packaging==25.0 ninja==1.13.0
```

Install the matching PyTorch build first, then install the recorded direct
dependencies. FlashAttention is built against the active PyTorch environment,
so build isolation is disabled deliberately.

```bash
uv pip install torch==2.7.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --no-build-isolation
uv pip install -e . --no-deps
```

The root `requirements.txt` is the canonical dependency list for reproducing
the released experiments.

## External requirements

- Download `Qwen3-4B-Instruct-2507` and set `MODEL_PATH` to its local path.
- Configure the judge and rubric-writer API through environment variables as
  described in the root README; do not store credentials in config files.
- Download the released annotations with `python scripts/download_data.py`.

## Verify

```bash
python - <<'PY'
import torch, transformers, vllm

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("gpu_count", torch.cuda.device_count())
PY

ruff check scripts rubric/openai_client.py \
  verl/experimental/reward_loop/reward_manager/rubric_scripts.py \
  verl/experimental/reward_loop/reward_manager/rubric_scripts_task.py
python -m pytest -q tests
```
