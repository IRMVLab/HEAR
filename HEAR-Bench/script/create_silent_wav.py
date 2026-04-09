import argparse
import os
import struct
import wave


def create_silent_wav(filename, duration_sec=10, sample_rate=16000, channels=1, samp_width=2):
    """Create a silent WAV file."""
    n_frames = int(duration_sec * sample_rate)
    comptype = "NONE"
    compname = "not compressed"

    if samp_width == 1:
        silent_value = 128
        fmt = "<B"
    else:
        silent_value = 0
        fmt = "<h"

    print(f"Creating file: {filename}")

    try:
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(samp_width)
            wav_file.setframerate(sample_rate)
            wav_file.setcomptype(comptype, compname)

            sample_data = struct.pack(fmt, silent_value)
            silent_frame = sample_data * channels
            all_frames_data = silent_frame * n_frames
            wav_file.writeframes(all_frames_data)

        print("Created silent WAV successfully.")
        print(f"  file: {os.path.abspath(filename)}")
        print(f"  duration: {duration_sec} s")
        print(f"  sample_rate: {sample_rate} Hz")
        print(f"  bit_depth: {samp_width * 8} bit")
        print(f"  channels: {channels}")
        print(f"  total_frames: {n_frames}")
    except Exception as exc:
        print(f"Failed to create WAV file: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Create a silent WAV file.")
    parser.add_argument("--output", type=str, default="silent_10_seconds.wav", help="Output WAV path")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--channels", type=int, default=1, help="Channel count")
    parser.add_argument("--sample-width", type=int, default=2, help="Sample width in bytes")
    args = parser.parse_args()

    create_silent_wav(
        args.output,
        duration_sec=args.duration,
        sample_rate=args.sample_rate,
        channels=args.channels,
        samp_width=args.sample_width,
    )


if __name__ == "__main__":
    main()
