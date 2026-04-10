# HEAR

HEAR is an audio-conditioned robot policy training project built on top of [openpi](https://github.com/Physical-Intelligence/openpi). This codebase keeps upstream openpi training, policy serving, and dataset tooling, and adds a Qwen3-Omni based PyTorch training path for sound-aware manipulation tasks.

This directory is intended to live as a subfolder of `IRMVLab/HEAR`, so all commands below assume:

```bash
cd hear
```

## What Is Included

- `openpi/`: core models, transforms, policies, training config, and serving code
- `train_pytorch_qwen3.py`: main HEAR training entrypoint
- `scripts/`: shared utilities such as norm-stat computation, JAX training, and policy serving
- `examples/`: dataset conversion, evaluation, and client examples
- `packages/openpi-client/`: lightweight websocket client for remote inference

## Requirements

- Ubuntu 22.04 or similar Linux environment
- Python 3.11
- NVIDIA GPU(s) with CUDA 12 support
- `uv` for dependency management
- `git-lfs` if you plan to work with large checkpoints or datasets

## Installation

HEAR reuses the upstream openpi environment setup, but the commands below assume this project lives inside `IRMVLab/HEAR/hear`.

```bash
git clone https://github.com/IRMVLab/HEAR.git
cd HEAR/hear

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` is still recommended because `lerobot` is installed as a dependency.

If you use ALOHA or LIBERO examples, clone their external dependencies into `hear/third_party/`:

```bash
git clone https://github.com/Physical-Intelligence/aloha.git third_party/aloha
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
```

## Transformers Patch

The Qwen3/PaliGemma path depends on patched Transformers files shipped in this repo:

```bash
cp -r ./openpi/models_pytorch/transformers_replace/* "$(python -c 'import transformers; print(transformers.__path__[0])')"
```

The project is currently tested against `transformers==4.53.2`.

## Model Assets

HEAR uses two kinds of assets:

1. Upstream openpi assets, for example `gs://openpi-assets/checkpoints/pi05_base`
2. HEAR-specific local or Hugging Face model folders for Qwen3-Omni, Qwen3-0.6B, and Mimi

Upstream openpi assets are still downloaded automatically through `openpi.shared.download`. By default they are cached in `~/.cache/openpi`; override that with `OPENPI_DATA_HOME`.

For HEAR-specific weights, the default local layout is:

```text
models/
  Qwen3-Omni-30B-A3B-Instruct-Pruned/
  Qwen3-Omni-30B-A3B-Thinking-Pruned/
  Qwen3-0.6B/
  mimi/
```

Recommended download flow after you publish the weights to Hugging Face:

```bash
huggingface-cli login
huggingface-cli download biubiu2/HEAR-Qwen3-Omni-30B-A3B-Instruct-Pruned --local-dir models/Qwen3-Omni-30B-A3B-Instruct-Pruned
huggingface-cli download biubiu2/HEAR-Qwen3-0.6B --local-dir models/Qwen3-0.6B
huggingface-cli download biubiu2/HEAR-mimi --local-dir models/mimi
```

If you store them elsewhere, set environment variables instead of editing code:

```bash
export HEAR_MODEL_HOME=/path/to/models
export HEAR_QWEN3_OMNI_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct-Pruned
export HEAR_QWEN3_OMNI_THINKING_PATH=/path/to/Qwen3-Omni-30B-A3B-Thinking-Pruned
export HEAR_QWEN3_PATH=/path/to/Qwen3-0.6B
export HEAR_AUDIO_CODEC_PATH=/path/to/mimi
```

## Training

The main HEAR training workflow is:

```bash
uv run scripts/compute_norm_stats.py --config-name robotwin_click_alarmclock_audio_random

uv run train_pytorch_qwen3.py robotwin_click_alarmclock_audio_random \
  --exp_name hear_release_run
```

The corresponding config lives in `openpi/training/config.py`. Checkpoints are written under:

```text
checkpoints/<config_name>/<exp_name>/
```

The code still supports upstream JAX/openpi training if you need it:

```bash
uv run scripts/train.py <config_name> --exp_name <run_name>
```

## Serving and Inference

Serve a trained checkpoint:

```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=robotwin_click_alarmclock_audio_random \
  --policy.dir=checkpoints/robotwin_click_alarmclock_audio_random/<exp_name>/<step>
```

The default server port is `8000`. For a lightweight client integration, install the bundled client package:

```bash
cd packages/openpi-client
uv pip install -e .
```

See `docs/remote_inference.md` and `examples/simple_client/README.md` for client usage.

## Dataset Conversion

Useful conversion scripts:

- `examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py`
- `examples/droid/convert_droid_data_to_lerobot_sanya.py`
- `examples/libero/convert_selfcollect_data_to_lerobot.py`
- `examples/libero/convert_labsim_data_to_lerobot.py`

All of them now expect explicit CLI arguments instead of hardcoded local paths.

## Publishing Guidance

Do not commit large pretrained weights or local checkpoints into this code repository. Publish them as separate Hugging Face model repos and keep the code repo lightweight. A practical layout is:

- `biubiu2/HEAR-Qwen3-Omni-30B-A3B-Instruct-Pruned`
- `biubiu2/HEAR-Qwen3-0.6B`
- `biubiu2/HEAR-mimi`
- optional separate repos or releases for HEAR fine-tuned checkpoints

If you keep the same directory names as the defaults above, users can download them into `hear/models/` and run the project without editing any source files.

## Notes

- The repo ignores `models/`, large safetensors files, checkpoints, and debug outputs by default.
- If you are offline or on a restricted cluster, pre-populate `OPENPI_DATA_HOME` with required upstream openpi assets.
- Some example READMEs still describe upstream openpi workflows; the commands in this README are the release-ready HEAR entrypoints.
- See `docs/release_checklist.md` for the manual publishing checklist and Hugging Face weight release flow.
