import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video, multi_camera_audio_to_video


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            if key == "audio" and len(value) > 0:
                audio_array = np.array(value)
                hdf5_group.create_dataset(key, data=audio_array)
                continue

            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            elif value and all(isinstance(v, (str, bytes)) for v in value):
                encoded = [
                    v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
                    for v in value
                ]
                str_arr = np.asarray(encoded, dtype=object)
                str_dtype = h5py.string_dtype(encoding="utf-8")
                hdf5_group.create_dataset(
                    key,
                    data=str_arr,
                    dtype=str_dtype,
                )
            else:
                hdf5_group.create_dataset(key, data=np.array(value))
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def pkl_files_to_hdf5_and_video(
    pkl_files,
    hdf5_path,
    video_path,
    render_mode="full",
    full_episode_audio=None,
    episode_audio_sample_rate=None,
    save_video=True,
):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    
    # Preserve full-episode audio when available.
    episode_audio = (None if full_episode_audio is None else np.array(full_episode_audio, dtype=np.float32, copy=True))
    
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)
        
        if episode_audio is None and isinstance(pkl_file.get("full_episode_audio"), np.ndarray):
            episode_audio = np.array(pkl_file["full_episode_audio"], dtype=np.float32, copy=True)

    has_audio = "audio" in data_list and len(data_list["audio"]) > 0

    has_multi_camera = (
        "observation" in data_list and
        "left_camera" in data_list["observation"] and
        "right_camera" in data_list["observation"] and
        len(data_list["observation"]["left_camera"].get("rgb", [])) > 0 and
        len(data_list["observation"]["right_camera"].get("rgb", [])) > 0
    )
    
    has_qpos = (
        "joint_action" in data_list and
        "left_arm" in data_list["joint_action"] and
        "right_arm" in data_list["joint_action"] and
        len(data_list["joint_action"]["left_arm"]) > 0 and
        len(data_list["joint_action"]["right_arm"]) > 0
    )
    
    audio_sample_rate = episode_audio_sample_rate or 16000
    audio_status_dict = data_list.get("audio_status")
    if episode_audio_sample_rate is None and isinstance(audio_status_dict, dict):
        sr_list = audio_status_dict.get("sample_rate")
        if isinstance(sr_list, list) and len(sr_list) > 0:
            audio_sample_rate = sr_list[-1]
    if episode_audio is None and has_audio:
        fallback = data_list["audio"][-1] if len(data_list["audio"]) > 0 else None
        if fallback is not None:
            episode_audio = np.array(fallback, dtype=np.float32, copy=True)
            print("Warning: Missing recorder audio, fallback to last frame snippet.")
    has_episode_audio = episode_audio is not None and episode_audio.size > 0
    
    if save_video:
        if has_audio and has_multi_camera and has_episode_audio:
            audio_data = episode_audio
            print(f"Using full episode audio: {len(audio_data)} samples")
            
            save_freq = 1
            sim_timestep = 1/250

            if len(pkl_files) >= 2:
                file1_num = int(os.path.basename(pkl_files[0])[:-4])
                file2_num = int(os.path.basename(pkl_files[1])[:-4])
                save_freq = file2_num - file1_num

            left_qpos_data = np.array(data_list["joint_action"]["left_arm"]) if has_qpos else np.zeros((len(data_list["observation"]["head_camera"]["rgb"]), 6))
            right_qpos_data = np.array(data_list["joint_action"]["right_arm"]) if has_qpos else np.zeros((len(data_list["observation"]["head_camera"]["rgb"]), 6))
            
            print(f"Creating multi-camera video with audio (audio_length={len(audio_data)/audio_sample_rate:.2f}s, render_mode={render_mode})...")
            multi_camera_audio_to_video(
                head_imgs=np.array(data_list["observation"]["head_camera"]["rgb"]),
                left_imgs=np.array(data_list["observation"]["left_camera"]["rgb"]),
                right_imgs=np.array(data_list["observation"]["right_camera"]["rgb"]),
                audio_data=audio_data,
                audio_sample_rate=audio_sample_rate,
                left_qpos_data=left_qpos_data,
                right_qpos_data=right_qpos_data,
                out_path=video_path,
                sim_timestep=sim_timestep,
                save_freq=save_freq,
                render_mode=render_mode
            )
        else:
            print("Creating single camera video...")
            audio_kwargs = {}
            if has_episode_audio:
                audio_kwargs = {
                    "audio_data": episode_audio,
                    "audio_sample_rate": audio_sample_rate,
                }
            images_to_video(
                np.array(data_list["observation"]["head_camera"]["rgb"]),
                out_path=video_path,
                **audio_kwargs,
            )
    else:
        print("Skipping video creation (save_video=False).")

    # Save HDF5 metadata and attach full-episode audio when present.
    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)
        
        if episode_audio is not None and episode_audio.size > 0:
            dset = f.create_dataset("full_episode_audio", data=episode_audio)
            dset.attrs["sample_rate"] = audio_sample_rate
            print(f"Saved full episode audio: {len(episode_audio)} samples")


def process_folder_to_hdf5_video(
    folder_path,
    hdf5_path,
    video_path,
    render_mode="full",
    full_episode_audio=None,
    audio_sample_rate=None,
    save_video=True,
):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(
        pkl_files,
        hdf5_path,
        video_path,
        render_mode=render_mode,
        full_episode_audio=full_episode_audio,
        episode_audio_sample_rate=audio_sample_rate,
        save_video=save_video,
    )
