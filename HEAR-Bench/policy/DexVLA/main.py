import argparse
import os
import pickle

import torch
from safetensors import safe_open


def compare_lora_weights(path):
    ckpt = safe_open(os.path.join(path, "adapter_model.safetensors"), framework="pt")
    ema_ckpt = safe_open(os.path.join(path, "ema", "adapter_model.safetensors"), framework="pt")

    for k in ckpt.keys():
        print(k, torch.equal(ckpt.get_tensor(k), ema_ckpt.get_tensor(k)))


def compare_non_lora_weights(path):
    ckpt = torch.load(os.path.join(path, "non_lora_trainables.bin"))
    try:
        ema_ckpt = torch.load(os.path.join(path, "ema_non_lora_trainables.bin"))
    except Exception as exc:
        print(exc)
        ema_ckpt = torch.load(os.path.join(path, "ema", "non_lora_trainables.bin"))

    for k in ckpt.keys():
        print(k, torch.equal(ckpt[k], ema_ckpt[k]))


def compare_zero_weights(path, tag="global_step30000"):
    ckpt = torch.load(
        os.path.join(path, tag, "bf16_zero_pp_rank_6_mp_rank_00_optim_states.pt"),
        map_location=torch.device("cpu"),
    )["optimizer_state_dict"]
    ema_ckpt = torch.load(
        os.path.join(path, "ema", tag, "bf16_zero_pp_rank_6_mp_rank_00_optim_states.pt"),
        map_location=torch.device("cpu"),
    )["optimizer_state_dict"]
    print(ckpt.keys())
    for k in ckpt.keys():
        print(k, torch.equal(ckpt[k], ema_ckpt[k]))


def compare_ema_weights(path):
    ckpt = torch.load(os.path.join(path, "non_lora_trainables.bin"), map_location=torch.device("cpu"))
    ema_ckpt = torch.load(os.path.join(path, "ema_weights_trainable.pth"), map_location=torch.device("cpu"))
    for k in ema_ckpt.keys():
        if "policy_head" in k:
            bool_matrix = ckpt[k] == ema_ckpt[k]
            false_indices = torch.where(bool_matrix == False)
            print(k, bool_matrix, false_indices)
            for i, j in zip(false_indices[0], false_indices[1]):
                print(ckpt[k].shape, ckpt[k][i][j].to(ema_ckpt[k].dtype).item(), ema_ckpt[k][i][j].item())
            break
        if k in ckpt.keys():
            print(k, ckpt[k].dtype, ema_ckpt[k].dtype, torch.equal(ckpt[k].to(ema_ckpt[k].dtype), ema_ckpt[k]))
        else:
            print(f"no weights for {k} in ckpt")


def check_norm_stats(stats_path):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    gripper = {}
    for k, v in stats.items():
        gripper[k] = {}
        for kk, vv in v.items():
            gripper[k][kk] = [vv[6], vv[13]]
    return gripper


def parse_args():
    parser = argparse.ArgumentParser(description="DexVLA checkpoint comparison utility.")
    parser.add_argument(
        "--mode",
        choices=["compare-lora", "compare-non-lora", "compare-zero", "compare-ema", "check-norm"],
        required=True,
    )
    parser.add_argument("--path", help="Checkpoint directory used by comparison modes.")
    parser.add_argument("--tag", default="global_step30000", help="Optimizer state tag for compare-zero mode.")
    parser.add_argument("--stats-path", help="Path to dataset_stats.pkl for check-norm mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "check-norm":
        if not args.stats_path:
            raise ValueError("--stats-path is required when --mode=check-norm")
        print(check_norm_stats(args.stats_path))
    else:
        if not args.path:
            raise ValueError("--path is required for checkpoint comparison modes")
        if args.mode == "compare-lora":
            compare_lora_weights(args.path)
        elif args.mode == "compare-non-lora":
            compare_non_lora_weights(args.path)
        elif args.mode == "compare-zero":
            compare_zero_weights(args.path, args.tag)
        elif args.mode == "compare-ema":
            compare_ema_weights(args.path)
