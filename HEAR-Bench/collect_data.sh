#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

if [ -n "${gpu_id}" ]; then
  export CUDA_VISIBLE_DEVICES=${gpu_id}
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py --task_name "${task_name}" --task_config "${task_config}"
rm -rf data/${task_name}/${task_config}/.cache
