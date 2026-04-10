"""
Script to convert LabSim dataset to LeRobot format.

LabSim dataset uses HDF5 format containing camera images, robot joint angles and action data.
This script converts it to LeRobot standard format.

Usage:
python scripts/convert_labsim_data_to_lerobot.py --data_dir /path/to/your/labsim/dataset --num_processes 4

To push to Hugging Face Hub:
python scripts/convert_labsim_data_to_lerobot.py --data_dir /path/to/your/labsim/dataset --push_to_hub --num_processes 8

Note: This script requires LeRobot installation:
pip install lerobot
"""

import os
import h5py
import numpy as np
import tyro
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import queue
import time

# Try to import LeRobot modules
try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed, please run: pip install lerobot")

RAW_DATASET_NAMES = [
    "07.17.40_Level1_pick/dataset",
    # "07.18.56_Level1_pour/dataset",
    # "07.23.31_Level1_shake/dataset",
    # "07.25.32_Level1_stir/dataset",
    # "07.27.04_Level1_press/dataset",
    # "07.35.32_Level1_open/dataset",
] 

def get_image_shape_from_h5(h5_file: h5py.File, camera_name: str) -> tuple:
    """Get image shape from HDF5 file (directly from root, not from groups)"""
    if camera_name in h5_file.keys():
        dataset = h5_file[camera_name]
        if hasattr(dataset, 'shape'):
            if len(dataset.shape) == 4:  # [T, H, W, C]
                return dataset.shape[1:]  # Return [H, W, C]
            elif len(dataset.shape) == 3:  # [H, W, C]
                return dataset.shape
    return (256, 256, 3)  # Default shape


def get_state_shape_from_h5(h5_file: h5py.File) -> tuple:
    """Get state data shape from HDF5 file (directly from root)"""
    if "agent_pose" in h5_file.keys():
        dataset = h5_file["agent_pose"]
        if hasattr(dataset, 'shape'):
            if len(dataset.shape) == 2:  # [T, num_joints]
                return (dataset.shape[1],)  # Return [num_joints]
            elif len(dataset.shape) == 1:  # [num_joints]
                return (dataset.shape[0],)
    return (7,)  # Default joint count


def get_action_shape_from_h5(h5_file: h5py.File) -> tuple:
    """Get action data shape from HDF5 file (directly from root)"""
    if "actions" in h5_file.keys():
        dataset = h5_file["actions"]
        if hasattr(dataset, 'shape'):
            if len(dataset.shape) == 2:  # [T, num_joints]
                return (dataset.shape[1],)  # Return [num_joints]
            elif len(dataset.shape) == 1:  # [num_joints]
                return (dataset.shape[0],)
    return (7,)  # Default joint count


def detect_camera_names(h5_file: h5py.File) -> list:
    """Detect camera names MACRO HDF5 file (directly from root)"""
    camera_names = []
    for data_key in h5_file.keys():
        if data_key not in ["agent_pose", "actions", "language_instruction"]:
            # Check if it's image data (by shape)
            dataset = h5_file[data_key]
            if hasattr(dataset, 'shape') and len(dataset.shape) >= 3:
                # Assume 3D or 4D data is image
                if data_key not in camera_names:
                    camera_names.append(data_key)
    return camera_names


def get_source_fps_from_h5(h5_file: h5py.File) -> int:
    """Detect source fps from HDF5 file metadata or estimate from data"""
    # Try to get fps from metadata first
    if 'fps' in h5_file.attrs:
        return int(h5_file.attrs['fps'])
    
    # Estimate fps from episode duration and frame count
    if 'duration' in h5_file.attrs:
        duration = h5_file.attrs['duration']
        # Find frame count from any camera data
        for data_key in h5_file.keys():
            if data_key not in ["agent_pose", "actions", "language_instruction"]:
                dataset = h5_file[data_key]
                if hasattr(dataset, 'shape') and len(dataset.shape) >= 3:
                    frame_count = dataset.shape[0] if len(dataset.shape) == 4 else 1
                    if duration > 0:
                        estimated_fps = frame_count / duration
                        return int(estimated_fps)
    
    return 60  # Default assumption


def check_language_instructions_availability(episode_files: list) -> tuple:
    """Check if language instructions are available in the dataset"""
    episodes_with_instructions = 0
    total_episodes = len(episode_files)
    
    for episode_file in episode_files:
        try:
            with h5py.File(episode_file, 'r') as h5_file:
                if "language_instruction" in h5_file:
                    episodes_with_instructions += 1
        except Exception:
            pass
    
    return episodes_with_instructions, total_episodes


def calculate_frame_indices(source_fps: int, target_fps: int, total_frames: int) -> list:
    """Calculate which frames to keep when converting fps"""
    if source_fps <= target_fps:
        # If source fps is lower or equal, keep all frames
        return list(range(total_frames))
    
    # Calculate frame interval
    interval = source_fps / target_fps
    
    # Generate frame indices to keep
    frame_indices = []
    current_frame = 0
    
    while current_frame < total_frames:
        frame_indices.append(int(current_frame))
        current_frame += interval
    
    return frame_indices


def process_episode(args):
    """Process a single episode file - designed to be run in parallel"""
    episode_file, camera_names, source_fps, target_fps, image_shape, state_shape, action_shape = args
    
    try:
        frame_data_list = []
        episode_name = Path(episode_file).stem  # Get episode_0001 from episode_0001.h5
        
        with h5py.File(episode_file, 'r') as h5_file:
            # Get time steps from root level
            if camera_names and camera_names[0] in h5_file:
                time_steps = h5_file[camera_names[0]].shape[0] if len(h5_file[camera_names[0]].shape) == 4 else 1
            elif "agent_pose" in h5_file:
                time_steps = h5_file["agent_pose"].shape[0] if len(h5_file["agent_pose"].shape) == 2 else 1
            else:
                return None, f"No valid data found in episode {episode_name}"
            
            # Get language instruction if available (from root level)
            language_instruction = None
            if "language_instruction" in h5_file:
                try:
                    instruction_data = h5_file["language_instruction"]
                    if hasattr(instruction_data, 'asstr'):
                        language_instruction = instruction_data.asstr()[()]
                    elif hasattr(instruction_data, 'decode'):
                        language_instruction = instruction_data.decode('utf-8')
                    else:
                        language_instruction = instruction_data[()]
                        if isinstance(language_instruction, bytes):
                            language_instruction = language_instruction.decode('utf-8')
                        elif isinstance(language_instruction, np.ndarray):
                            language_instruction = str(language_instruction.item())
                        else:
                            language_instruction = str(language_instruction)
                    print(f"Found language instruction for episode {episode_name}: {language_instruction}")
                except Exception as e:
                    print(f"Warning: Could not decode language instruction for episode {episode_name}: {e}")
            
            # Calculate which frames to keep based on fps conversion
            frame_indices = calculate_frame_indices(source_fps, target_fps, time_steps)
            
            # Process each frame
            for t in frame_indices:
                frame_data = {}
                
                # Add camera data (from root level)
                for camera_name in camera_names:
                    if camera_name in h5_file:
                        camera_data = h5_file[camera_name]
                        if len(camera_data.shape) == 4:  # [T, H, W, C]
                            frame_data[camera_name] = camera_data[t]
                        else:  # [H, W, C]
                            frame_data[camera_name] = camera_data
                
                # Add state data (from root level)
                if "agent_pose" in h5_file:
                    pose_data = h5_file["agent_pose"]
                    if len(pose_data.shape) == 2:  # [T, num_joints]
                        frame_data["state"] = pose_data[t]
                    else:  # [num_joints]
                        frame_data["state"] = pose_data
                
                # Add action data (from root level)

                if "actions" in h5_file:
                    action_data = h5_file["actions"]
                    if len(action_data.shape) == 2:  # [T, num_joints]
                        frame_data["actions"] = action_data[t]
                    else:  # [num_joints]
                        frame_data["actions"] = action_data
                
                # Set task description - use language instruction if available, otherwise use episode name
                if language_instruction:
                    frame_data["task"] = language_instruction
                else:
                    frame_data["task"] = f"LabSim episode {episode_name}"
                
                frame_data_list.append(frame_data)
        
        return frame_data_list, None
    
    except Exception as e:
        return None, f"Error processing episode {episode_name}: {str(e)}"


def main(data_dir: str, repo_name: str, *, push_to_hub: bool = False, fps: int = 60, robot_type: str = "franka", num_processes: int = 8, output_dir: Optional[str] = None):
    """Main conversion function
    
    Args:
        data_dir: Path to the LabSim dataset root directory containing multiple dataset subdirectories
        repo_name: Name for the output dataset
        push_to_hub: Whether to push to Hugging Face Hub
        fps: Target fps for conversion
        robot_type: Type of robot (default: franka)
        num_processes: Number of processes for parallel processing (default: 8)
        output_dir: Optional custom output directory (default: uses HF_LEROBOT_HOME)
    """
    if not LEROBOT_AVAILABLE:
        print("Error: LeRobot not installed, cannot continue conversion")
        return
    
    # Validate num_processes
    if num_processes < 1:
        num_processes = 1
    max_processes = mp.cpu_count()
    if num_processes > max_processes:
        print(f"Warning: Requested {num_processes} processes, but only {max_processes} CPUs available. Using {max_processes} processes.")
        num_processes = max_processes

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory does not exist: {data_path}")
        return

    # Collect all episode files from all datasets
    all_episode_files = []
    dataset_info = []  # Store (dataset_name, episode_files) pairs
    
    print(f"Scanning {len(RAW_DATASET_NAMES)} datasets...")
    for raw_dataset_name in RAW_DATASET_NAMES:
        dataset_path = data_path / raw_dataset_name
        if not dataset_path.exists():
            print(f"Warning: Dataset path does not exist: {dataset_path}, skipping...")
            continue
        
        # Find all episode_*.h5 files in this dataset
        episode_files = sorted(list(dataset_path.glob("episode_*.h5")))
        if not episode_files:
            print(f"Warning: No episode_*.h5 files found in {dataset_path}, skipping...")
            continue
        
        print(f"Found {len(episode_files)} episode files in {raw_dataset_name}")
        all_episode_files.extend(episode_files)
        dataset_info.append((raw_dataset_name, episode_files))
    
    if not all_episode_files:
        print(f"Error: No episode_*.h5 files found in any dataset")
        return

    print(f"Total {len(all_episode_files)} episode files across {len(dataset_info)} datasets")

    # Use first episode file to determine data structure
    first_episode_file = all_episode_files[0]
    print(f"Reading dataset structure from: {first_episode_file}")
    
    with h5py.File(first_episode_file, 'r') as h5_file:
        # Detect camera names
        camera_names = detect_camera_names(h5_file)
        print(f"Detected cameras: {camera_names}")
        
        # Check language instructions availability
        episodes_with_instructions, total_episodes = check_language_instructions_availability(all_episode_files)
        print(f"Language instructions: {episodes_with_instructions}/{total_episodes} episodes have language instructions")

        # Get data shapes
        if camera_names:
            image_shape = get_image_shape_from_h5(h5_file, camera_names[0])
        else:
            image_shape = (256, 256, 3)
        
        state_shape = get_state_shape_from_h5(h5_file)
        action_shape = get_action_shape_from_h5(h5_file)
        
        # Detect source fps
        source_fps = get_source_fps_from_h5(h5_file)
        print(f"Source fps: {source_fps}, Target fps: {fps}")
        
        print(f"Image shape: {image_shape}")
        print(f"State shape: {state_shape}")
        print(f"Action shape: {action_shape}")

    # Determine output path
    if output_dir:
        output_path = Path(output_dir) / repo_name
    else:
        output_path = HF_LEROBOT_HOME / repo_name
    
    # Clean output directory
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"Cleaned existing output directory: {output_path}")

    # Create LeRobot dataset
    features = {}

    # Add camera features
    for camera_name in camera_names:
        features[camera_name] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    
    # Add state and action features
    features["state"] = {
        "dtype": "float32",
        "shape": state_shape,
        "names": ["state"],
    }
    
    features["actions"] = {
        "dtype": "float32",
        "shape": action_shape,
        "names": ["actions"],
    }

    print("Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=8,
    )

    # Loop over raw LabSim datasets and write episodes to the LeRobot dataset
    # Similar to how libero script handles multiple datasets
    print(f"Converting data using {num_processes} processes...")
    
    episode_count = 0
    for raw_dataset_name, episode_files in dataset_info:
        print(f"\nProcessing dataset: {raw_dataset_name} ({len(episode_files)} episodes)")
        
        # Prepare arguments for parallel processing
        process_args = [
            (episode_file, camera_names, source_fps, fps, image_shape, state_shape, action_shape)
            for episode_file in episode_files
        ]
        
        # Use multiprocessing to process episodes in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all tasks
            future_to_episode = {executor.submit(process_episode, args): args[0] for args in process_args}
            
            # Process results as they complete
            for future in future_to_episode:
                episode_file = future_to_episode[future]
                episode_name = Path(episode_file).stem
                try:
                    frame_data_list, error = future.result()
                    if error:
                        print(f"Error: {error}")
                        continue
                    
                    print(f"Processing episode: {episode_name} ({len(frame_data_list)} frames)")
                    
                    # Add frames to dataset sequentially (thread-safe)
                    for frame_data in frame_data_list:
                        dataset.add_frame(frame_data)
                    
                    # Save episode
                    dataset.save_episode()
                    episode_count += 1
                    
                except Exception as e:
                    print(f"Exception processing episode {episode_name}: {str(e)}")
                    continue

    print(f"\nSuccessfully converted {episode_count} episodes from {len(dataset_info)} datasets")

    # Optional: Push to Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["labsim", robot_type, "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Dataset pushed to Hugging Face Hub")
        print(f"Dataset available at: https://huggingface.co/datasets/{repo_name}")
    else:
        print(f"Dataset saved locally to: {output_path}")
        print(f"You can load it later using: LeRobotDataset('{repo_name}', ...)")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Ensure proper multiprocessing on all platforms
    tyro.cli(main) 
