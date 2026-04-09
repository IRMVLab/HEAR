import argparse
from pathlib import Path
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sample_rate


def list_output_devices() -> list[tuple[int, dict]]:
    devices = sd.query_devices()
    output_devices = [(idx, dev) for idx, dev in enumerate(devices) if dev["max_output_channels"] > 0]
    print("Available output devices:")
    for idx, dev in output_devices:
        print(f"[{idx:02d}] {dev['name']} | output channels: {dev['max_output_channels']}")
    if not output_devices:
        print("No output devices were detected.")
    print()
    return output_devices


def test_devices(samples: np.ndarray, sample_rate: int, devices: list[tuple[int, dict]]):
    for idx, dev in devices:
        print(f"Testing device [{idx}] {dev['name']} ...")
        resp = ""
        try:
            sd.stop()
            sd.play(samples, samplerate=sample_rate, device=idx, blocking=False)
            resp = input("Press Enter for next device, y to confirm, or q to quit: ").strip().lower()
            sd.stop()
            print("Playback stopped.")
        except Exception as exc:
            print(f"Device [{idx}] failed: {exc}")
            continue
        if resp == "q":
            break
        if resp == "y":
            print("Confirmed working device. Stopping early.")
            break
        print()
    print("Device test finished.")


def build_default_audio_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "assets" / "audios" / "padding_audio.mp3"


def main():
    parser = argparse.ArgumentParser(description="Play a test clip on each output device.")
    parser.add_argument(
        "--audio",
        type=Path,
        default=build_default_audio_path(),
        help="Audio file to play (default: assets/audios/padding_audio.mp3)",
    )
    args = parser.parse_args()

    try:
        samples, sample_rate = load_audio(args.audio)
    except Exception as exc:
        print(f"Failed to load audio: {exc}")
        sys.exit(1)

    devices = list_output_devices()
    if not devices:
        sys.exit(1)

    test_devices(samples, sample_rate, devices)


if __name__ == "__main__":
    if sd is None:
        print("sounddevice is unavailable. Install it first or run in an environment with audio support.")
        sys.exit(1)
    main()
