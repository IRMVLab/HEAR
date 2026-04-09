import argparse
import os
import sys

import cv2
import h5py
import numpy as np

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    print("Error: scipy is required to export WAV files.")
    print("Install it with: pip install scipy")
    sys.exit(1)


def decode_and_save_images(image_dataset, output_dir):
    """Decode a JPEG-padded HDF5 image dataset and save each frame as JPG."""
    print(f"  Extracting {len(image_dataset)} image frames to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(image_dataset)):
        encoded_data_padded = image_dataset[i]
        encoded_data = encoded_data_padded.rstrip(b"\0")

        if not encoded_data:
            print(f"    Warning: frame {i} is empty and was skipped.")
            continue

        np_arr = np.frombuffer(encoded_data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            print(f"    Warning: failed to decode frame {i}. Skipping.")
            continue

        img_path = os.path.join(output_dir, f"{i:04d}.jpg")
        cv2.imwrite(img_path, img)


def find_and_extract_images(h5_group, output_base_path):
    """Recursively locate `rgb` datasets and export them as JPG frames."""
    for key, value in h5_group.items():
        if isinstance(value, h5py.Group):
            current_output_path = os.path.join(output_base_path, key)
            find_and_extract_images(value, current_output_path)
        elif isinstance(value, h5py.Dataset) and key == "rgb":
            decode_and_save_images(value, output_base_path)


def _prepare_audio_for_wav(audio_data):
    """Convert audio data into a dtype supported by scipy.io.wavfile.write."""
    if audio_data.dtype == np.float64:
        audio_data = audio_data.astype(np.float32)

    if audio_data.dtype == np.float32:
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val == 0:
            audio_data = np.zeros_like(audio_data, dtype=np.float32)
    elif audio_data.dtype not in (np.int16, np.uint8):
        max_abs_val = np.max(np.abs(audio_data))
        if max_abs_val == 0:
            audio_data = np.zeros_like(audio_data, dtype=np.int16)
        elif max_abs_val > 32767:
            audio_data = (audio_data / max_abs_val * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)

    return audio_data


def _get_sample_rate(h5_file, default_sr=16000):
    if "full_episode_audio" in h5_file:
        sr = h5_file["full_episode_audio"].attrs.get("sample_rate")
        if sr is not None:
            return int(sr)
    if "audio_status/sample_rate" in h5_file:
        try:
            sr_list = h5_file["audio_status/sample_rate"][:]
            if len(sr_list) > 0:
                return int(sr_list[-1])
        except Exception:
            pass
    return default_sr


def extract_full_audio(h5_file, output_folder):
    """Extract full-episode audio and save it as one WAV file."""
    print("Extracting full-episode audio...")
    audio_data = None
    sample_rate = _get_sample_rate(h5_file)

    if "full_episode_audio" in h5_file:
        audio_data = h5_file["full_episode_audio"][:]
        print("  Found full_episode_audio.")
    elif "audio" in h5_file:
        print("  full_episode_audio not found. Falling back to chunked audio.")
        audio_chunks = h5_file["audio"][:]
        if audio_chunks.ndim == 2:
            audio_data = audio_chunks.flatten()
            print(f"  Flattened audio chunks (length: {len(audio_data)}).")
        else:
            print(f"  audio dataset is not 2D (shape: {audio_chunks.shape}). Using it as-is.")
            audio_data = audio_chunks
    else:
        print("  No audio dataset found. Skipping full-audio export.")
        return

    print(f"  Sample rate: {sample_rate} Hz")
    audio_data = _prepare_audio_for_wav(audio_data)

    output_path = os.path.join(output_folder, "extracted_audio.wav")
    try:
        wavfile.write(output_path, sample_rate, audio_data)
        print(f"  Saved full audio to: {output_path}")
    except Exception as exc:
        print(f"  Failed to write full WAV file: {exc}")
        print("  Audio stats:")
        print(
            f"  dtype: {audio_data.dtype}, shape: {audio_data.shape}, "
            f"min: {np.min(audio_data)}, max: {np.max(audio_data)}"
        )


def extract_frame_audio(h5_file, output_folder, sample_rate=None):
    """Extract per-frame audio chunks and save each chunk as an individual WAV file."""
    print("\nExtracting per-frame audio...")

    if "audio" not in h5_file:
        print("  audio dataset not found. Skipping per-frame export.")
        return

    audio_dataset = h5_file["audio"]
    if audio_dataset.ndim != 2:
        print(f"  audio dataset is not 2D (shape: {audio_dataset.shape}). Skipping.")
        return

    num_frames, samples_per_frame = audio_dataset.shape
    sr = sample_rate or _get_sample_rate(h5_file)
    frame_audio_dir = os.path.join(output_folder, "frame_audio")
    os.makedirs(frame_audio_dir, exist_ok=True)

    print(f"  Found {num_frames} frames, {samples_per_frame} samples per frame.")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Output dir: {frame_audio_dir}")

    for i in range(num_frames):
        audio_data = _prepare_audio_for_wav(audio_dataset[i])
        output_path = os.path.join(frame_audio_dir, f"frame_{i:04d}.wav")
        try:
            wavfile.write(output_path, sr, audio_data)
        except Exception as exc:
            print(f"  Failed to write frame {i}: {exc}")

    print(f"  Exported {num_frames} frame audio files.")


def main():
    parser = argparse.ArgumentParser(description="Extract images and audio from an HDF5 episode file.")
    parser.add_argument(
        "--hdf5_file",
        type=str,
        default="data/click_alarmclock_audio_single/demo_clean_audio/data/episode0.hdf5",
        help="Input HDF5 file path.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output/extracted_data",
        help="Output directory for extracted assets.",
    )
    args = parser.parse_args()

    input_path = args.hdf5_file
    output_path = args.output_folder

    if not os.path.exists(input_path):
        print(f"Error: HDF5 file not found: {input_path}")
        sys.exit(1)

    os.makedirs(output_path, exist_ok=True)
    print(f"Processing {input_path}")
    print(f"Writing outputs to {output_path}")

    with h5py.File(input_path, "r") as h5_file:
        extract_full_audio(h5_file, output_path)
        extract_frame_audio(h5_file, output_path)

        print("\nExtracting images...")
        if "observation" in h5_file:
            find_and_extract_images(h5_file["observation"], os.path.join(output_path, "observation"))
        else:
            print("  observation group not found. Searching from the file root.")
            find_and_extract_images(h5_file, output_path)

    print("\nExtraction finished.")


if __name__ == "__main__":
    main()
