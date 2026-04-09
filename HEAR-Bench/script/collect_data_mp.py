import datetime
import importlib
import json
import math
import os
import shutil
import sys
import traceback
from argparse import ArgumentParser

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml


current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append("./")


def setup_worker_logging(_rank):
    """Force line-buffered stdout so logs flush under nohup."""
    sys.stdout.reconfigure(line_buffering=True)


def log(rank, message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][W-{rank}] {message}", flush=True)


def class_decorator(task_name):
    """Instantiate a task environment by module name."""
    try:
        envs_module = importlib.import_module(f"envs.{task_name}")
        env_class = getattr(envs_module, task_name)
        return env_class()
    except ImportError as exc:
        print(f"Error loading task module: {exc}")
        raise SystemExit(f"Could not import envs.{task_name}") from exc
    except Exception as exc:
        print(f"Error instantiating task {task_name}: {exc}")
        traceback.print_exc()
        raise SystemExit("Task instantiation failed") from exc


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def worker_search_seeds(rank, gpu_id, task_name, task_config, args, search_start_seed, max_attempts, target_count):
    del task_config
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    setup_worker_logging(rank)
    logical_gpu = torch.cuda.current_device() if torch.cuda.is_available() else "N/A"
    log(rank, f"Seed search started | physical GPU: {gpu_id} | logical CUDA: {logical_gpu}")
    log(rank, f"Target seeds: {target_count} | start seed: {search_start_seed}")

    try:
        task_env = class_decorator(task_name)
    except Exception as exc:
        log(rank, f"Fatal: failed to initialize environment: {exc}")
        return

    found_seeds = []
    current_seed = search_start_seed

    for attempt in range(max_attempts):
        if len(found_seeds) >= target_count:
            break

        try:
            setup_args = args.copy()
            setup_args["need_plan"] = True

            task_env.setup_demo(now_ep_num=0, seed=current_seed, **setup_args)
            task_env.play_once()

            if task_env.plan_success and task_env.check_success():
                log(rank, f"Found successful seed: {current_seed}")
                task_env.save_traj_data(current_seed)
                found_seeds.append(current_seed)

            task_env.close_env()
        except Exception:
            try:
                task_env.close_env()
            except Exception:
                pass

        current_seed += 1

        if attempt % 50 == 0 and attempt > 0:
            log(rank, f"Still searching... tried up to {current_seed}, found {len(found_seeds)}")

    part_file = os.path.join(args["save_path"], f"seed_part_{rank}.txt")
    with open(part_file, "w", encoding="utf-8") as f:
        for seed in found_seeds:
            f.write(f"{seed} ")

    log(rank, "Seed search finished.")


def worker_collect_data(rank, gpu_id, task_name, task_config, args, sub_seed_list, start_idx):
    del task_config
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    setup_worker_logging(rank)
    log(rank, f"Collection started | physical GPU: {gpu_id} | episodes: {len(sub_seed_list)}")

    try:
        task_env = class_decorator(task_name)
    except Exception as exc:
        log(rank, f"Fatal: failed to initialize environment: {exc}")
        return

    local_info_db = {}
    clear_cache_freq = args.get("clear_cache_freq", 10)

    for i, seed in enumerate(sub_seed_list):
        current_episode_idx = start_idx + i
        try:
            run_args = args.copy()
            run_args["need_plan"] = False
            run_args["render_freq"] = 0
            run_args["save_data"] = True

            task_env.setup_demo(now_ep_num=current_episode_idx, seed=seed, **run_args)

            traj_data = task_env.load_tran_data(current_episode_idx)
            run_args["left_joint_path"] = traj_data["left_joint_path"]
            run_args["right_joint_path"] = traj_data["right_joint_path"]
            task_env.set_path_lst(run_args)

            info = task_env.play_once()
            local_info_db[f"episode_{current_episode_idx}"] = info

            should_clear = ((i + 1) % clear_cache_freq == 0)
            task_env.close_env(clear_cache=should_clear)
            task_env.merge_pkl_to_hdf5_video()
            task_env.remove_data_cache()

            if not task_env.check_success():
                log(rank, f"Episode {current_episode_idx} (seed {seed}) failed success check.")
            elif i % 5 == 0:
                log(rank, f"Finished episode {current_episode_idx} (seed {seed})")
        except Exception as exc:
            log(rank, f"Error on episode {current_episode_idx}: {exc}")
            try:
                task_env.close_env()
            except Exception:
                pass

    part_file = os.path.join(args["save_path"], f"scene_info_part_{rank}.json")
    with open(part_file, "w", encoding="utf-8") as f:
        json.dump(local_info_db, f, ensure_ascii=False, indent=4)

    log(rank, "Collection finished.")


def run_parallel_seed_generation(args, num_workers, gpu_ids):
    target_total = args["episode_num"]
    traj_dir = os.path.join(args["save_path"], "_traj_data")
    os.makedirs(traj_dir, exist_ok=True)

    seed_file_path = os.path.join(args["save_path"], "seed.txt")
    current_seeds = []
    if os.path.exists(seed_file_path):
        with open(seed_file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                current_seeds = [int(x) for x in content.split()]

    needed = target_total - len(current_seeds)
    if needed <= 0:
        return len(current_seeds)

    print("\n" + "=" * 40, flush=True)
    print(f" >>> Phase 1: parallel seed completion (missing {needed})", flush=True)
    print("=" * 40, flush=True)

    quota_per_worker = math.ceil(needed / num_workers)
    search_range_per_worker = 2000

    processes = []
    start_seed_base = max(current_seeds) + 1 if current_seeds else 0

    for rank in range(num_workers):
        gpu_id = gpu_ids[rank % len(gpu_ids)]
        worker_start_seed = start_seed_base + rank * search_range_per_worker

        process = mp.Process(
            target=worker_search_seeds,
            args=(
                rank,
                gpu_id,
                args["task_name"],
                args["task_config"],
                args,
                worker_start_seed,
                search_range_per_worker,
                quota_per_worker,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print("Merging cached trajectory data...", flush=True)
    new_found_seeds = []

    for rank in range(num_workers):
        part_file = os.path.join(args["save_path"], f"seed_part_{rank}.txt")
        if os.path.exists(part_file):
            with open(part_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    new_found_seeds.extend(int(x) for x in content.split())
            os.remove(part_file)

    valid_new_seeds = []
    current_idx_start = len(current_seeds)

    for i, seed in enumerate(new_found_seeds):
        target_idx = current_idx_start + i
        if target_idx >= target_total:
            break

        src_pkl = os.path.join(traj_dir, f"episode{seed}.pkl")
        dst_pkl = os.path.join(traj_dir, f"episode{target_idx}.pkl")

        if not os.path.exists(src_pkl):
            print(f"Warning: seed {seed} was recorded but its trajectory file is missing. Skipping.", flush=True)
            continue

        if os.path.exists(dst_pkl):
            os.remove(dst_pkl)

        try:
            os.rename(src_pkl, dst_pkl)
            valid_new_seeds.append(seed)
        except Exception as exc:
            print(f"Rename failed: {exc}", flush=True)

    final_seeds = current_seeds + valid_new_seeds
    with open(seed_file_path, "w", encoding="utf-8") as f:
        for seed in final_seeds:
            f.write(f"{seed} ")

    print(f"Seed preparation complete: {len(final_seeds)} / {target_total}", flush=True)
    return len(final_seeds)


def run_parallel_data_collection(args, num_workers, gpu_ids):
    print("\n" + "=" * 40, flush=True)
    print(f" >>> Phase 2: parallel data collection (workers: {num_workers})", flush=True)
    print("=" * 40, flush=True)

    seed_path = os.path.join(args["save_path"], "seed.txt")
    if not os.path.exists(seed_path):
        print("Error: seed.txt was not found.", flush=True)
        return

    with open(seed_path, "r", encoding="utf-8") as f:
        seed_list = [int(x) for x in f.read().split()]

    target_num = args["episode_num"]
    if len(seed_list) > target_num:
        seed_list = seed_list[:target_num]

    print(f"Planned episodes: {len(seed_list)}", flush=True)

    chunks = np.array_split(seed_list, num_workers)
    processes = []
    start_counter = 0

    for rank in range(num_workers):
        sub_seeds = chunks[rank].tolist()
        if not sub_seeds:
            continue

        gpu_id = gpu_ids[rank % len(gpu_ids)]
        process = mp.Process(
            target=worker_collect_data,
            args=(rank, gpu_id, args["task_name"], args["task_config"], args, sub_seeds, start_counter),
        )
        process.start()
        processes.append(process)
        start_counter += len(sub_seeds)

    for process in processes:
        process.join()

    print("Merging scene_info.json ...", flush=True)
    final_info_db = {}

    info_file_path = os.path.join(args["save_path"], "scene_info.json")
    if os.path.exists(info_file_path):
        try:
            with open(info_file_path, "r", encoding="utf-8") as f:
                final_info_db = json.load(f)
        except Exception:
            pass

    for rank in range(num_workers):
        part_file = os.path.join(args["save_path"], f"scene_info_part_{rank}.json")
        if os.path.exists(part_file):
            try:
                with open(part_file, "r", encoding="utf-8") as f:
                    final_info_db.update(json.load(f))
                os.remove(part_file)
            except Exception as exc:
                print(f"Failed to merge scene info part {rank}: {exc}", flush=True)

    with open(info_file_path, "w", encoding="utf-8") as file:
        json.dump(final_info_db, file, ensure_ascii=False, indent=4)

    desc_dir = os.path.join(project_root, "description")
    if os.path.exists(desc_dir):
        try:
            cmd = (
                f"cd {desc_dir} && bash gen_episode_instructions.sh "
                f"{args['task_name']} {args['task_config']} {args['language_num']}"
            )
            print(f"Generating instructions: {cmd}", flush=True)
            os.system(cmd)
        except Exception as exc:
            print(f"Instruction generation failed: {exc}", flush=True)
    else:
        print(f"description directory not found: {desc_dir}. Skipping instruction generation.", flush=True)

    print("All phases completed.", flush=True)


def main(task_name, task_config, num_workers, gpu_ids):
    config_path = os.path.join(project_root, "task_config", f"{task_config}.yml")
    if not os.path.exists(config_path):
        config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    os.makedirs(args["save_path"], exist_ok=True)

    configs_path = os.path.join(project_root, "task_config")
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(configs_path, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(config_name):
        path = embodiment_types[config_name]["file_path"]
        if not path.startswith("/"):
            path = os.path.join(project_root, path)
        return path

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    seed_path = os.path.join(args["save_path"], "seed.txt")
    current_seeds_count = 0
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            current_seeds_count = len(f.read().split())

    need_collect_seeds = (not args["use_seed"]) or (current_seeds_count < args["episode_num"])

    if need_collect_seeds:
        print(
            f"Seed generation required (target: {args['episode_num']}, current: {current_seeds_count})",
            flush=True,
        )
        loop_cnt = 0
        while current_seeds_count < args["episode_num"]:
            run_parallel_seed_generation(args, num_workers, gpu_ids)
            if os.path.exists(seed_path):
                with open(seed_path, "r", encoding="utf-8") as f:
                    current_seeds_count = len(f.read().split())
            loop_cnt += 1
            if loop_cnt > 10:
                print("Warning: still not enough seeds after repeated attempts. Stopping search.", flush=True)
                break
    else:
        print("Skipping seed generation (use_seed=True and enough seeds already exist).", flush=True)

    if args["collect_data"]:
        if not os.path.exists(seed_path) or os.path.getsize(seed_path) == 0:
            print("Error: seed file is missing or empty. Cannot start collection.", flush=True)
            return
        run_parallel_data_collection(args, num_workers, gpu_ids)
    else:
        print("collect_data is disabled in the task config. Exiting.", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("--task_name", type=str, default="touch_plate_metal")
    parser.add_argument("--task_config", type=str, default="demo_clean_audio")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of worker processes")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs; omit to auto-detect")

    args_parsed = parser.parse_args()

    if args_parsed.gpus:
        gpu_list = [int(x.strip()) for x in args_parsed.gpus.split(",") if x.strip()]
    elif torch.cuda.is_available():
        gpu_list = list(range(torch.cuda.device_count()))
        print(f"Auto-detected GPUs: {gpu_list}", flush=True)
    else:
        gpu_list = [0]

    main(
        task_name=args_parsed.task_name,
        task_config=args_parsed.task_config,
        num_workers=args_parsed.num_workers,
        gpu_ids=gpu_list,
    )

    try:
        cache_dir = os.path.join("data", args_parsed.task_name, args_parsed.task_config, ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception:
        pass
