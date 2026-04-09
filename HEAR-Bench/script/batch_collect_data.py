import argparse
import subprocess
import sys


DEFAULT_TASKS = [
    ("adjust_bottle_reset", "demo_clean_audio"),
    ("show_bottle_yes", "demo_clean_audio_100"),
    ("touch_plate_plastic", "demo_clean_audio_50"),
    ("touch_plate_metal", "demo_clean_audio_50"),
]


def parse_tasks(raw_tasks):
    tasks = []
    for item in raw_tasks:
        if ":" not in item:
            raise ValueError(f"Invalid task spec '{item}'. Expected task_name:task_config.")
        task_name, task_config = item.split(":", 1)
        tasks.append((task_name.strip(), task_config.strip()))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run multi-process data collection for a list of tasks.")
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Task spec in the form task_name:task_config. Can be provided multiple times.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Worker count passed to collect_data_mp.py")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable")
    parser.add_argument(
        "--collect-script",
        type=str,
        default="script/collect_data_mp.py",
        help="Collector entrypoint path",
    )
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks) if args.tasks else DEFAULT_TASKS

    for task_name, task_config in tasks:
        cmd = [
            args.python,
            args.collect_script,
            "--task_name",
            task_name,
            "--task_config",
            task_config,
            "--num_workers",
            str(args.num_workers),
            "--gpus",
            args.gpus,
        ]
        print(f"Collecting: {task_name} | {task_config}")
        subprocess.run(cmd, check=True)
        print(f"Finished: {task_name} | {task_config}\n")


if __name__ == "__main__":
    main()
