# HEAR Release Checklist

## Before Publishing

- Rotate any Hugging Face token that has been pasted into chat, shell history, or local notes.
- Keep pretrained weights and fine-tuned checkpoints out of the code repository.
- Verify that `models/`, `checkpoints/`, `debug_data/`, and large `.safetensors` files remain ignored by git.

## Recommended Hugging Face Layout

Publish large assets as separate model repos instead of bundling them into `hear/`:

- `biubiu2/HEAR-Qwen3-Omni-30B-A3B-Instruct-Pruned`
- `biubiu2/HEAR-Qwen3-Omni-30B-A3B-Thinking-Pruned` if you still need the thinking variant
- `biubiu2/HEAR-Qwen3-0.6B`
- `biubiu2/HEAR-mimi`
- optional: `biubiu2/HEAR-checkpoints` for released HEAR checkpoints

The local defaults in `openpi/training/config.py` expect these folders under `hear/models/`. If you use different names, set:

```bash
export HEAR_QWEN3_OMNI_PATH=/path/to/your/omni
export HEAR_QWEN3_OMNI_THINKING_PATH=/path/to/your/omni-thinking
export HEAR_QWEN3_PATH=/path/to/your/qwen3-0.6b
export HEAR_AUDIO_CODEC_PATH=/path/to/your/mimi
```

## Upload Commands

For large folders, prefer the current Hugging Face CLI large-folder uploader:

```bash
hf auth login
hf upload-large-folder --repo-type model biubiu2/HEAR-Qwen3-Omni-30B-A3B-Instruct-Pruned ./models/Qwen3-Omni-30B-A3B-Instruct-Pruned
hf upload-large-folder --repo-type model biubiu2/HEAR-Qwen3-0.6B ./models/Qwen3-0.6B
hf upload-large-folder --repo-type model biubiu2/HEAR-mimi ./models/mimi
```

For downloads, the matching command is:

```bash
hf download biubiu2/HEAR-Qwen3-0.6B --local-dir models/Qwen3-0.6B
```

## Repository Integration

This project is prepared to live under:

```text
IRMVLab/HEAR/
  hear/
```

When you copy it into the parent repository:

- run commands from `IRMVLab/HEAR/hear`
- keep model folders under `IRMVLab/HEAR/hear/models`
- clone optional external dependencies into `IRMVLab/HEAR/hear/third_party`

## Final Validation

- `python -m py_compile` on the modified training and utility scripts
- dry-run installation in a fresh environment with `uv sync && uv pip install -e .`
- one end-to-end smoke test:
  - `uv run scripts/compute_norm_stats.py --config-name robotwin_click_alarmclock_audio_random`
  - `uv run train_pytorch_qwen3.py robotwin_click_alarmclock_audio_random --exp_name smoke_test`
  - `uv run scripts/serve_policy.py policy:checkpoint --policy.config=robotwin_click_alarmclock_audio_random --policy.dir=<checkpoint_dir>`
