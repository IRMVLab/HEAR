task_name=${1}
setting=${2}
max_num=${3}

python utils/generate_episode_instructions.py --task_name "$task_name" --setting "$setting" --max_num "$max_num"
