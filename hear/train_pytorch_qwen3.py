"""
PyTorch training entrypoint for the HEAR Qwen3-based policy.

Usage
Single GPU:
  uv run train_pytorch_qwen3.py <config_name> --exp_name <run_name>
Multi-GPU:
  uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> train_pytorch_qwen3.py <config_name> --exp_name <run_name>
"""

import dataclasses
import gc
import logging
import os
from pathlib import Path
import sys
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb
import torch.nn.functional as F

import openpi.models.pi0_config
from openpi.models_pytorch.pi0_pytorch_qwen3 import PI0Pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data

def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)

def _wandb_safe_dict(obj):
    if isinstance(obj, dict):
        return {str(k): _wandb_safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_wandb_safe_dict(v) for v in obj]
    return obj

def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=_wandb_safe_dict(dataclasses.asdict(config)),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
        sampling_weights_config=config.sampling_weights_config,
    )
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def get_latest_checkpoint_dir(base_checkpoint_dir):
    """Find the latest checkpoint directory in the base checkpoint path.
    
    Args:
        base_checkpoint_dir: Base directory containing experiment checkpoint folders
        
    Returns:
        Path to the latest checkpoint directory, or None if not found
    """
    if not base_checkpoint_dir.exists():
        return None
    
    # Find all directories that contain checkpoint subdirectories (numeric folders)
    checkpoint_dirs = []
    for exp_dir in base_checkpoint_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
            # Check if this directory contains numeric checkpoint folders
            has_checkpoints = any(
                d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
                for d in exp_dir.iterdir()
            )
            if has_checkpoints:
                # Get the latest checkpoint step in this experiment directory
                latest_step = get_latest_checkpoint_step(exp_dir)
                if latest_step is not None:
                    checkpoint_dirs.append((exp_dir, latest_step, exp_dir.stat().st_mtime))
    
    if not checkpoint_dirs:
        return None
    
    # Sort by modification time (newest first) and return the directory
    checkpoint_dirs.sort(key=lambda x: x[2], reverse=True)
    return checkpoint_dirs[0][0]


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def evaluate_model(model, data_loader, device, num_eval_batches=10, num_steps=10):
    """Run a short evaluation pass on the training data loader."""
    model_to_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model_to_eval.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    total_accurate_samples = 0
    total_first_accurate_samples = 0
    num_samples = 0
    accuracy_threshold = 0.005

    with torch.no_grad():
        eval_iter = iter(data_loader)

        for batch_idx in range(num_eval_batches):
            try:
                observation, gt_actions = next(eval_iter)
            except StopIteration:
                break

            observation = jax.tree.map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
                observation,
            )
            gt_actions = gt_actions.to(device)

            pred_actions = model_to_eval.sample_actions(
                device=device,
                observation=observation,
                noise=None,
                num_steps=num_steps
            )

            batch_size = gt_actions.shape[0]

            mse = F.mse_loss(pred_actions, gt_actions, reduction="sum").item()
            mae = F.l1_loss(pred_actions, gt_actions, reduction="sum").item()

            total_mse += mse
            total_mae += mae
            num_samples += batch_size * gt_actions.shape[1] * gt_actions.shape[2]

            sample_mse = F.mse_loss(pred_actions, gt_actions, reduction="none")
            sample_mse = sample_mse.mean(dim=[1, 2])
            accurate_samples = (sample_mse < accuracy_threshold).sum().item()
            total_accurate_samples += accurate_samples
            total_samples += batch_size

            first_pred = pred_actions[:, 0, :]
            first_gt = gt_actions[:, 0, :]
            first_mse = F.mse_loss(first_pred, first_gt, reduction="none")
            first_mse = first_mse.mean(dim=1)
            first_accurate_samples = (first_mse < accuracy_threshold).sum().item()
            total_first_accurate_samples += first_accurate_samples

    model_to_eval.train()

    avg_mse = total_mse / num_samples if num_samples > 0 else 0.0
    avg_mae = total_mae / num_samples if num_samples > 0 else 0.0
    avg_rmse = np.sqrt(avg_mse)
    accuracy = total_accurate_samples / total_samples if total_samples > 0 else 0.0
    first_action_accuracy = total_first_accurate_samples / total_samples if total_samples > 0 else 0.0
    
    return {
        "eval/mse": avg_mse,
        "eval/mae": avg_mae,
        "eval/rmse": avg_rmse,
        "eval/accuracy": accuracy,
        "eval/first_action_accuracy": first_action_accuracy,
        "eval/num_samples": num_samples,
        "eval/accuracy_threshold": accuracy_threshold,
    }


def train_loop(config: _config.TrainConfig):
    """Main training loop for the HEAR PyTorch trainer."""
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    resuming = False
    if config.resume:
        if config.pytorch_weight_path:
            resume_ckpt_dir = Path(config.pytorch_weight_path)
            if not resume_ckpt_dir.exists():
                raise FileNotFoundError(f"pytorch_weight_path {resume_ckpt_dir} does not exist.")
            if (resume_ckpt_dir / "model.safetensors").exists():
                checkpoint_root = resume_ckpt_dir.parent
                step_name = resume_ckpt_dir.name
                latest_step = int(step_name) if step_name.isdigit() else get_latest_checkpoint_step(checkpoint_root)
                logging.info(
                    f"Detected specific checkpoint step directory: {resume_ckpt_dir} "
                    f"(parsed step: {latest_step})"
                )
            else:
                checkpoint_root = resume_ckpt_dir
                latest_step = get_latest_checkpoint_step(checkpoint_root)
            if latest_step is None:
                raise FileNotFoundError(f"No valid checkpoints found in {resume_ckpt_dir}")
            resuming = True
            config = dataclasses.replace(
                config,
                exp_name=checkpoint_root.name,
            )
            logging.info(
                f"Resuming from pytorch_weight_path: {resume_ckpt_dir} "
                f"at step {latest_step} (checkpoint root: {checkpoint_root})"
            )
        else:
            base_checkpoint_dir = config.checkpoint_dir.parent
            exp_checkpoint_dir = get_latest_checkpoint_dir(base_checkpoint_dir)
            
            if exp_checkpoint_dir is not None:
                latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
                if latest_step is not None:
                    resuming = True
                    config = dataclasses.replace(config, exp_name=exp_checkpoint_dir.name)
                    logging.info(
                        f"Resuming from latest checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                    )
                else:
                    raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir}")
            else:
                raise FileNotFoundError(
                    f"No checkpoint directories found in {base_checkpoint_dir} for resume. "
                    f"Please ensure there are existing experiment directories with checkpoints."
                )
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    exp_checkpoint_dir = config.checkpoint_dir
    if not resuming:
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        logging.info(f"Using existing experiment checkpoint directory: {exp_checkpoint_dir}")

    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    loader, data_config = build_datasets(config)

    if is_main and config.wandb_enabled and not resuming:
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        images_to_log = []
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = PI0Pytorch(model_cfg, config).to(device)

    # Optional PI0 checkpoint wrappers; Qwen3 per-layer GC is handled in Qwen3WithExpertModel.
    enable_pi0_checkpointing = bool(getattr(config, "pi0_gradient_checkpointing", True))
    if enable_pi0_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    enable_qwen3_layer_gc = True
    enable_gc_for_ddp = enable_pi0_checkpointing or enable_qwen3_layer_gc

    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

        if enable_gc_for_ddp:
            model._set_static_graph()
            logging.info("Set static graph for DDP to work with gradient checkpointing")

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None and not resuming:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path, strict=False
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(
            f"Memory optimizations: qwen3_layer_gc={enable_qwen3_layer_gc}, "
            f"pi0_checkpointing={enable_pi0_checkpointing}"
        )
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    eval_interval = config.eval_interval
    num_eval_batches = config.num_eval_batches

    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            observation = jax.tree.map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
                observation,
            )
            actions = actions.to(torch.float32)
            actions = actions.to(device)

            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            losses = model(observation, actions)

            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            loss.backward()

            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            optim.step()
            optim.zero_grad(set_to_none=True)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []

            global_step += 1

            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            if is_main and global_step > 0 and global_step % eval_interval == 0:
                logging.info(f"Running evaluation at step {global_step}...")
                eval_start_time = time.time()

                eval_metrics = evaluate_model(
                    model=model,
                    data_loader=loader,
                    device=device,
                    num_eval_batches=num_eval_batches,
                )

                eval_time = time.time() - eval_start_time
                eval_metrics["eval/time_seconds"] = eval_time

                logging.info(
                    f"Evaluation at step {global_step}: "
                    f"MSE={eval_metrics['eval/mse']:.6f}, "
                    f"MAE={eval_metrics['eval/mae']:.6f}, "
                    f"RMSE={eval_metrics['eval/rmse']:.6f}, "
                    f"ACC={eval_metrics['eval/accuracy']:.4f}, "
                    f"FirstACC={eval_metrics['eval/first_action_accuracy']:.4f}, "
                    f"time={eval_time:.2f}s"
                )

                if config.wandb_enabled:
                    wandb.log(eval_metrics, step=global_step)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    if is_main:
        logging.info("Running final evaluation...")
        final_eval_metrics = evaluate_model(
            model=model,
            data_loader=loader,
            device=device,
            num_eval_batches=num_eval_batches * 2,
        )

        logging.info(
            f"Final evaluation: "
            f"MSE={final_eval_metrics['eval/mse']:.6f}, "
            f"MAE={final_eval_metrics['eval/mae']:.6f}, "
            f"RMSE={final_eval_metrics['eval/rmse']:.6f}, "
            f"ACC={final_eval_metrics['eval/accuracy']:.4f}, "
            f"FirstACC={final_eval_metrics['eval/first_action_accuracy']:.4f}"
        )
        
        if config.wandb_enabled:
            wandb.log(final_eval_metrics, step=global_step)

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()

    if len(sys.argv) <= 1:
        sys.argv.insert(1, os.environ.get("OPENPI_CONFIG_NAME", "robotwin_click_alarmclock_audio_random"))

    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
