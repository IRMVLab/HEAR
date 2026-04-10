"""
Script to convert self-collected dataset to LeRobot format.

The self-collected dataset uses HDF5 format containing compressed camera images, 
robot joint angles, action data, and audio data.
This script converts it to LeRobot standard format.

Usage:
python scripts/convert_selfcollect_data_to_lerobot.py --data_dir /path/to/your/dataset --num_processes 4

To push to Hugging Face Hub:
python scripts/convert_selfcollect_data_to_lerobot.py --data_dir /path/to/your/dataset --push_to_hub --num_processes 8

Note: This script requires LeRobot installation:
pip install lerobot
"""

import os
import h5py
import numpy as np
import tyro
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import cv2

# Try to import LeRobot modules
try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed, please run: pip install lerobot")

RAW_DATASET_NAMES = [
    "./",
]


def decompress_image_data(compressed_data: bytes, height: int, width: int, channels: int = 3) -> np.ndarray:
    """
    Decompress JPEG compressed image data
    
    Args:
        compressed_data: JPEG compressed bytes
        height: Target image height
        width: Target image width
        channels: Number of channels (3 for RGB, 1 for depth)
    
    Returns:
        np.ndarray: Decompressed image array
    """
    if channels == 3:
        # RGB image
        img = cv2.imdecode(np.frombuffer(compressed_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to zeros if decompression fails
            return np.zeros((height, width, channels), dtype=np.uint8)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Depth image (grayscale)
        img = cv2.imdecode(np.frombuffer(compressed_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((height, width), dtype=np.uint8)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
    
    # Resize if needed
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    
    return img


def load_camera_images(h5_file: h5py.File, camera_suffix: str, frame_idx: int, 
                       rgb_height: int = 256, rgb_width: int = 256) -> Dict[str, np.ndarray]:
    """
    Load RGB and depth images for a specific camera at a given frame
    
    Args:
        h5_file: HDF5 file object
        camera_suffix: Camera suffix ('', '_r', '_l' for main, right, left)
        frame_idx: Frame index to load
        rgb_height: RGB image height
        rgb_width: RGB image width
    
    Returns:
        dict: Dictionary with 'rgb' and optionally 'depth' keys
    """
    result = {}
    
    # Load RGB data
    rgb_data_key = f'rgb_data{camera_suffix}'
    rgb_sizes_key = f'rgb_sizes{camera_suffix}'
    
    if rgb_data_key in h5_file and rgb_sizes_key in h5_file:
        rgb_sizes = h5_file[rgb_sizes_key][:]
        rgb_data = h5_file[rgb_data_key][:]
        
        # Calculate offset for this frame
        offset = int(np.sum(rgb_sizes[:frame_idx]))
        size = int(rgb_sizes[frame_idx])
        
        # Extract and decompress
        compressed_rgb = bytes(rgb_data[offset:offset + size])
        rgb_img = decompress_image_data(compressed_rgb, rgb_height, rgb_width, channels=3)
        result['rgb'] = rgb_img
    
    # Load depth data (optional)
    depth_data_key = f'depth_data{camera_suffix}'
    depth_sizes_key = f'depth_sizes{camera_suffix}'
    
    if depth_data_key in h5_file and depth_sizes_key in h5_file:
        depth_sizes = h5_file[depth_sizes_key][:]
        depth_data = h5_file[depth_data_key][:]
        
        offset = int(np.sum(depth_sizes[:frame_idx]))
        size = int(depth_sizes[frame_idx])
        
        compressed_depth = bytes(depth_data[offset:offset + size])
        depth_img = decompress_image_data(compressed_depth, rgb_height, rgb_width, channels=1)
        result['depth'] = depth_img
    
    return result


def detect_camera_configs(h5_file: h5py.File) -> List[str]:
    """
    Detect available camera configurations in the HDF5 file
    
    Returns:
        List of camera suffixes: ['', '_r', '_l'] for available cameras
    """
    camera_suffixes = []
    
    # Check for main camera
    if 'rgb_data' in h5_file and 'rgb_sizes' in h5_file:
        camera_suffixes.append('')
    
    # Check for right camera
    if 'rgb_data_r' in h5_file and 'rgb_sizes_r' in h5_file:
        camera_suffixes.append('_r')
    
    # Check for left camera
    if 'rgb_data_l' in h5_file and 'rgb_sizes_l' in h5_file:
        camera_suffixes.append('_l')
    
    return camera_suffixes


def get_camera_name_from_suffix(suffix: str) -> str:
    """Convert camera suffix to camera name used in features"""
    if suffix == '':
        return 'camera_1_rgb'
    elif suffix == '_r':
        return 'image_r'
    elif suffix == '_l':
        return 'image_l'
    return f'image{suffix}'


def get_image_shape_from_h5(h5_file: h5py.File, camera_suffix: str = '') -> tuple:
    """Get image shape from first frame"""
    try:
        images = load_camera_images(h5_file, camera_suffix, 0)
        if 'rgb' in images:
            return images['rgb'].shape
    except:
        pass
    return (256, 256, 3)  # Default shape


def get_state_shape_from_h5(h5_file: h5py.File) -> tuple:
    """Get state data shape from HDF5 file"""
    if "proprio" in h5_file.keys():
        dataset = h5_file["proprio"]
        if hasattr(dataset, 'shape'):
            if len(dataset.shape) == 2:
                return (dataset.shape[1],)
            elif len(dataset.shape) == 1:
                return (dataset.shape[0],)
    return (8,)  # Default: 7 joints + 1 gripper


def get_audio_shape_from_h5(h5_file: h5py.File) -> tuple:
    """Get audio data shape from HDF5 file"""
    if "audio" in h5_file.keys():
        dataset = h5_file["audio"]
        if hasattr(dataset, 'shape'):
            if len(dataset.shape) == 2:
                # audio shape: (n_frames, n_samples)
                return (dataset.shape[1],)
            elif len(dataset.shape) == 1:
                return (dataset.shape[0],)
    return (80000,)  # Default: 5 seconds at 16kHz


def get_audio_samplerate_from_h5(h5_file: h5py.File) -> int:
    """Get audio sample rate from HDF5 file metadata"""
    meta = h5_file.get('meta', {})
    if hasattr(meta, 'attrs') and 'audio_samplerate' in meta.attrs:
        return int(meta.attrs['audio_samplerate'])
    return 16000  # Default sample rate


def detect_action_key(h5_file: h5py.File) -> Optional[str]:
    """Detect whether the HDF5 file uses 'actions' or 'action' (prefer 'actions')."""
    if 'actions' in h5_file:
        return 'actions'
    if 'action' in h5_file:
        return 'action'
    return None


def get_action_shape_from_h5(h5_file: h5py.File) -> tuple:
    """Get action data shape from HDF5 file. Supports both 'actions' and 'action'."""
    action_key = detect_action_key(h5_file)
    if action_key is not None:
        dataset = h5_file[action_key]
        if hasattr(dataset, 'shape'):
            if len(dataset.shape) == 2:
                return (dataset.shape[1],)
            elif len(dataset.shape) == 1:
                return (dataset.shape[0],)
    # Fallback: check proprio shape
    return get_state_shape_from_h5(h5_file)


def process_episode(args):
    """Process a single episode file - designed to be run in parallel"""
    episode_file, camera_suffixes, source_fps, target_fps, image_shape, state_shape, action_shape, audio_shape = args
    
    try:
        frame_data_list = []
        episode_name = Path(episode_file).stem
        
        with h5py.File(episode_file, 'r') as h5_file:
            # Get time steps from proprio data
            if "proprio" in h5_file:
                time_steps = h5_file["proprio"].shape[0]
            else:
                return None, f"No proprio data found in episode {episode_name}"
            
            # Get language instruction (task_instruction)
            language_instruction = None
            if "task_instruction" in h5_file:
                try:
                    instruction_data = h5_file["task_instruction"]
                    if hasattr(instruction_data, 'asstr'):
                        language_instruction = instruction_data.asstr()[()]
                    else:
                        language_instruction = instruction_data[()]
                        if isinstance(language_instruction, bytes):
                            language_instruction = language_instruction.decode('utf-8')
                        elif isinstance(language_instruction, np.ndarray):
                            language_instruction = str(language_instruction.item())
                        else:
                            language_instruction = str(language_instruction)
                    print(f"Found task instruction for episode {episode_name}: {language_instruction}")
                except Exception as e:
                    print(f"Warning: Could not decode task instruction for episode {episode_name}: {e}")
            
            # Load action data (supports both 'actions' and 'action', prefer 'actions')
            action_data = None
            action_key = detect_action_key(h5_file)
            if action_key is not None:
                action_data = h5_file[action_key]
                # for debug/info
                print(f"Episode {episode_name}: Using action key '{action_key}' with shape {getattr(action_data, 'shape', None)}")
            
            # Load audio data (if exists)
            audio_data = None
            if "audio" in h5_file:
                audio_data = h5_file["audio"]
                print(f"Episode {episode_name}: Found audio data with shape {audio_data.shape}")
            
            # No fps conversion needed for filtered data
            frame_indices = list(range(time_steps))
            
            # Process each frame
            for t in frame_indices:
                frame_data = {}
                
                # Add camera data (RGB images)
                for camera_suffix in camera_suffixes:
                    images = load_camera_images(h5_file, camera_suffix, t, 
                                               rgb_height=image_shape[0], 
                                               rgb_width=image_shape[1])
                    
                    # Add RGB image with simplified key name
                    if 'rgb' in images:
                        camera_name = get_camera_name_from_suffix(camera_suffix)
                        frame_data[camera_name] = images['rgb']
                
                # Add state data (proprio) - ensure float32
                if "proprio" in h5_file:
                    proprio_data = h5_file["proprio"]
                    frame_data["state"] = np.array(proprio_data[t], dtype=np.float32)
                
                # Add action data - ensure float32. Output key always "actions"
                if action_data is not None:
                    frame_data["actions"] = np.array(action_data[t], dtype=np.float32)
                else:
                    # Fallback: if no action data, use next frame's proprio
                    proprio_data = h5_file["proprio"]
                    if t < time_steps - 1:
                        frame_data["actions"] = np.array(proprio_data[t + 1, :7], dtype=np.float32)
                    else:
                        frame_data["actions"] = np.array(proprio_data[t, :7], dtype=np.float32)
                
                # Add audio data (if exists) - ensure float32
                if audio_data is not None:
                    frame_data["audio"] = np.array(audio_data[t], dtype=np.float32)
                
                # Set task description
                if language_instruction:
                    frame_data["task"] = language_instruction
                else:
                    frame_data["task"] = f"Self-collected episode {episode_name}"
                
                frame_data_list.append(frame_data)
        
        return frame_data_list, None
    
    except Exception as e:
        import traceback
        return None, f"Error processing episode {episode_name}: {str(e)}\n{traceback.format_exc()}"


def main(data_dir: str, repo_name: str, *, push_to_hub: bool = False, fps: int = 10, robot_type: str = "franka", num_processes: int = 8, output_dir: Optional[str] = None):
    """Main conversion function
    
    Args:
        data_dir: Path to the self-collected dataset root directory containing multiple dataset subdirectories
        repo_name: Name for the output dataset
        push_to_hub: Whether to push to Hugging Face Hub
        fps: Target fps for the dataset (should match the filtered data fps)
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
    dataset_info = []
    
    print(f"Scanning {len(RAW_DATASET_NAMES)} datasets...")
    for raw_dataset_name in RAW_DATASET_NAMES:
        dataset_path = data_path / raw_dataset_name
        if not dataset_path.exists():
            print(f"Warning: Dataset path does not exist: {dataset_path}, skipping...")
            continue
        
        # Find all *.h5 files in this dataset
        episode_files = sorted(list(dataset_path.glob("*.h5")))
        if not episode_files:
            print(f"Warning: No *.h5 files found in {dataset_path}, skipping...")
            continue
        
        print(f"Found {len(episode_files)} episode files in {raw_dataset_name}")
        all_episode_files.extend(episode_files)
        dataset_info.append((raw_dataset_name, episode_files))
    
    if not all_episode_files:
        print(f"Error: No *.h5 files found in any dataset")
        return

    print(f"Total {len(all_episode_files)} episode files across {len(dataset_info)} datasets")

    # Use first episode file to determine data structure
    first_episode_file = all_episode_files[0]
    print(f"Reading dataset structure from: {first_episode_file}")
    
    with h5py.File(first_episode_file, 'r') as h5_file:
        # Detect camera configurations
        camera_suffixes = detect_camera_configs(h5_file)
        camera_names = [get_camera_name_from_suffix(s) for s in camera_suffixes]
        print(f"Detected cameras: {camera_names}")

        # Get data shapes
        if camera_suffixes:
            image_shape = get_image_shape_from_h5(h5_file, camera_suffixes[0])
        else:
            image_shape = (256, 256, 3)
        
        state_shape = get_state_shape_from_h5(h5_file)
        action_shape = get_action_shape_from_h5(h5_file)
        audio_shape = get_audio_shape_from_h5(h5_file)
        audio_samplerate = get_audio_samplerate_from_h5(h5_file)
        
        # Detect source fps
        # source_fps = get_source_fps_from_h5(h5_file)
        source_fps = 10
        print(f"Source fps: {source_fps}")
        
        print(f"Image shape: {image_shape}")
        print(f"State shape: {state_shape}")
        print(f"Action shape: {action_shape}")
        print(f"Audio shape: {audio_shape}")
        print(f"Audio sample rate: {audio_samplerate}Hz")
        
        # Check if audio data exists
        has_audio = "audio" in h5_file

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
    
    # Add audio feature (if audio data exists)
    if has_audio:
        features["audio"] = {
            "dtype": "float32",
            "shape": audio_shape,
            "names": ["audio"],
        }
        print(f"Added audio feature with shape {audio_shape}")

    print("Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=8,
    )

    # Loop over datasets and write episodes
    print(f"Converting data using {num_processes} processes...")
    
    episode_count = 0
    for raw_dataset_name, episode_files in dataset_info:
        print(f"\nProcessing dataset: {raw_dataset_name} ({len(episode_files)} episodes)")
        
        # Prepare arguments for parallel processing
        process_args = [
            (episode_file, camera_suffixes, source_fps, fps, image_shape, state_shape, action_shape, audio_shape)
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
            tags=["self-collected", robot_type, "hdf5", "audio"],
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
    mp.set_start_method("spawn", force=True)
    tyro.cli(main)
