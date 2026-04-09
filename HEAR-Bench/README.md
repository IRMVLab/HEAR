# HEAR-Bench

HEAR-Bench is an audio-augmented benchmark built on top of RoboTwin 2.0. This project focuses on two workflows:

1. collecting simulation data with synchronized RGB, state, and audio observations
2. training and evaluating pi0-style policies on those tasks

The main HEAR-specific changes are in `envs/`, `assets/audios/`, the audio task configs in `task_config/`, and the pi0 data processing / evaluation pipeline.

## Scope

- Officially supported: data collection, audio task definitions, pi0 data processing, pi0 training, and pi0 evaluation.
- Main entrypoints:
  - collection: `script/collect_data.py`, `script/collect_data_mp.py`
  - processing: `policy/pi0/scripts/process_data.py`, `policy/pi0/scripts/process_data_mp.py`
  - evaluation: `script/eval_policy.py`
- Legacy compatibility wrappers `script/eval_policy_openpi_qwen3*.py` are kept, but the unified evaluation entry is `script/eval_policy.py`.

## Repository Layout

- `envs/`: RoboTwin tasks and HEAR audio task implementations
- `assets/audios/`: bundled HEAR-specific audio assets
- `task_config/`: collection/evaluation configs, including `demo_clean_audio*.yml`
- `script/`: collection, evaluation, asset installation, and utility scripts
- `policy/pi0/`: pi0 data processing, training, and deployment code
- `description/`: instruction generation used after collection

## Installation

HEAR-Bench follows the same simulator setup, dependency chain, and asset download flow as RoboTwin 2.0. Users should treat this repository as a RoboTwin-based extension rather than a separate simulator stack.

Reference:

- RoboTwin 2.0 install guide: <https://robotwin-platform.github.io/doc/usage/robotwin-install.html>
- RoboTwin 2.0 repository: <https://github.com/RoboTwin-Platform/RoboTwin>

If this project is added into `IRMVLab/HEAR`, keep the folder name as `HEAR-Bench` and run all commands from `HEAR/HEAR-Bench`.

### 1. Base RoboTwin environment

```bash
conda create -n hear-bench python=3.10 -y
conda activate hear-bench
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools ffmpeg
bash script/_install.sh
bash script/_download_assets.sh
```

If embodiment paths are still unresolved after downloading assets, run:

```bash
python script/update_embodiment_config_path.py
```

HEAR-specific audio files are already included in `assets/audios/`. No extra asset download is required for audio tasks beyond the standard RoboTwin assets.

### 2. pi0 environment

`policy/pi0` follows the OpenPI package layout and expects a separate Python 3.11 environment.

```bash
cd policy/pi0
pip install uv
uv sync
cd ../..
```

## Data Collection

Single-process collection:

```bash
bash collect_data.sh open_microwave_audio demo_clean_audio 0
```

Parallel collection:

```bash
python script/collect_data_mp.py \
  --task_name open_microwave_audio \
  --task_config demo_clean_audio \
  --gpus 0,1 \
  --num_workers 8
```

Collected trajectories are written to `data/<task_name>/<task_config>/`.

## Process Data for pi0

Sequential processing:

```bash
python policy/pi0/scripts/process_data.py \
  --task_name open_microwave_audio \
  --setting demo_clean_audio \
  --expert_data_num 50
```

Parallel processing:

```bash
python policy/pi0/scripts/process_data_mp.py \
  --task_name open_microwave_audio \
  --setting demo_clean_audio \
  --expert_data_num 50
```

By default, processed data is exported to `processed_data/<task>-<setting>-<num>`. Use `--output_dir` to override it.

## Train pi0

```bash
cd policy/pi0
bash finetune.sh robotwin hear_pi0 0
cd ../..
```

Use your processed dataset and OpenPI config as needed for custom experiments.

## Evaluate pi0

Recommended command:

```bash
python script/eval_policy.py \
  --config policy/pi0/deploy_policy.yml \
  --task_name pour_water_audio_full \
  --task_config demo_clean_audio \
  --policy_name pi0 \
  --train_config_name robotwin \
  --model_name hear_pi0 \
  --checkpoint_id 30000 \
  --ckpt_setting hear_pi0 \
  --seed 0
```

Evaluation outputs are saved under `eval_result/<task>/<policy>/<task_config>/<ckpt_setting>/`.

## Asset and Release Notes

- Upstream RoboTwin assets (`background_texture`, `embodiments`, `objects`) are still downloaded through `script/_download_assets.sh`.
- HEAR-Bench only adds lightweight audio assets under `assets/audios/`; users do not need a second asset package.
- When publishing into the `IRMVLab/HEAR` repository, this folder should be committed as `HEAR-Bench/`, and the parent repository README should link to this directory rather than duplicating setup details.

## Acknowledgement

HEAR-Bench builds on RoboTwin 2.0 and reuses its simulator, asset pipeline, and task infrastructure. Please cite RoboTwin and OpenPI as appropriate in downstream work.
