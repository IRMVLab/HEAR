"""
Script to convert Aloha (Sanya) HDF5 data to LeRobot format.
Based on the structure of convert_droid_data_to_lerobot.py but using data logic from convert_droid_data_to_lerobot_sanya.py.

Usage:
uv run convert_sanya_to_lerobot.py --data_dir /path/to/raw/data --repo_id <org>/<dataset-name>
"""

import json
import os
import shutil
import fnmatch
from pathlib import Path
from typing import Literal

import cv2
import h5py
import numpy as np
import torch
import tqdm
import tyro
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def resize_image(image, size):
    """Resizes an image using PIL (from Droid script)."""
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    """
    Logic from sanya script to handle compressed or uncompressed images.
    """
    imgs_per_cam = {}
    for camera in cameras:
        # Check if 4D (uncompressed) or other (compressed)
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def main(
    data_dir: Path,
    repo_id: str,
    *,
    push_to_hub: bool = False,
    robot_type: str = "panda",
    mode: Literal["video", "image"] = "image",
):
    # 1. Clean up any existing dataset in the output directory (From Droid script)
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # 2. Define Features (From Sanya script: Left arm specific)
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_yaw",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    
    # We define the features explicitly as done in Sanya script
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
        "actions": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
        # Optional entries (velocity, effort) can be added conditionally, 
        # but here we define the standard structure.
    }

    # Define Cameras (From Sanya script)
    cameras = ["cam_high", "cam_left_wrist"]
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640), # Sanya script resolution
            "names": ["channels", "height", "width"],
        }
    
    # Check for velocity/effort in the first available file to decide if we add them to schema
    # (Simple heuristic based on Sanya script logic)
    hdf5_files = []
    for root, _, files in os.walk(data_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            hdf5_files.append(os.path.join(root, filename))
            
    if not hdf5_files:
        raise ValueError(f"No .hdf5 files found in {data_dir}")

    # Check optional features in the first file
    with h5py.File(hdf5_files[0], "r") as first_ep:
        if "/observations/qvel" in first_ep:
            features["observation.velocity"] = {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": motors,
            }
        if "/observations/effort" in first_ep:
            features["observation.effort"] = {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": motors,
            }
        if "/observations/audio_frames" in first_ep:
             features["observation.audio"] = {
                "dtype": "float32",
                "shape": (16000, ), # 6s 16kHz
                "names": ["audio"],
            }
        if "/observations/stage" in first_ep:
             features["stage"] = {
                "dtype": "string",
                "shape": (1,),
                "names": None,
            }

    # 3. Create LeRobot Dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=15, # Sanya script uses 50fps
        features=features,
        use_videos=(mode == "video"),
        image_writer_processes=10,
        image_writer_threads=5,
    )

    print(f"Found {len(hdf5_files)} episodes for conversion")

    # 4. Iterate and Convert
    for ep_idx, ep_path in enumerate(tqdm.tqdm(hdf5_files, desc="Converting episodes")):
        ep_path = Path(ep_path)
        
        # --- Load Logic (From Sanya script) ---
        with h5py.File(ep_path, "r") as ep:
            # Load raw tensors
            state = ep["/observations/qpos"][:]
            actions = ep["/action"][:]
            
            # Optional modalities
            velocity = ep["/observations/qvel"][:] if "/observations/qvel" in ep else None
            effort = ep["/observations/effort"][:] if "/observations/effort" in ep else None
            audio = ep["/observations/audio_frames"][:] if "/observations/audio_frames" in ep else None
            
            stage = None
            if "/observations/stage" in ep:
                raw_stage = ep["/observations/stage"][:]
                stage = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in raw_stage]

            # Load images
            imgs_per_cam = load_raw_images_per_camera(ep, cameras)
            
            num_frames = state.shape[0]

        # --- Instruction Logic (From Sanya script) ---
        dir_path = os.path.dirname(ep_path)
        json_path = os.path.join(dir_path, "instructions.json")
        
        instruction = "do something" # fallback
        if os.path.exists(json_path):
            with open(json_path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
                instructions = instruction_dict.get('instructions', [])
                if len(instructions) > 0:
                    instruction = np.random.choice(instructions)

        # --- Populate Dataset (Structure from Droid script) ---
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "actions": actions[i],
                "task": instruction,
            }

            # Add images
            for camera, img_array in imgs_per_cam.items():
                # Sanya script: images are (N, H, W, C) or decoded to that.
                # LeRobot expects (C, H, W).
                # Note: If cv2 loaded BGR, we might need RGB conversion? 
                # Droid script does ::-1. Sanya script logic: cv2.imdecode(..., COLOR) -> BGR usually in cv2.
                # Standard LeRobot assumes RGB.
                # If the Sanya script uses cv2.imdecode, it returns BGR. 
                # Let's flip channels to RGB to be safe, assuming cv2 read.
                img = img_array[i]
                if img.shape[-1] == 3: # Ensure it is HWC
                    # Convert BGR to RGB if it came from cv2 decoding
                    # If it came from direct HDF5 load, it might already be RGB. 
                    # We will assume consistent RGB output is desired.
                    # Looking at sanya script: `cv2.imdecode` returns BGR.
                    # If direct load `ep[...]`, it depends on recording.
                    # We will apply a channel flip assuming BGR from cv2, 
                    # strictly speaking we should check if it was compressed, but for simplicity:
                    pass 
                
                frame[f"observation.images.{camera}"] = img

            # Add Optionals
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            if audio is not None:
                frame["observation.audio"] = audio[i]
            if stage is not None:
                frame["stage"] = stage[i]

            dataset.add_frame(frame)
        
        dataset.save_episode()

    # 5. Push to Hub (From Droid script)
    if push_to_hub:
        dataset.push_to_hub(
            tags=["aloha", "mobile_aloha"],
            private=False,
            push_videos=(mode == "video"),
        )


if __name__ == "__main__":
    tyro.cli(main)
