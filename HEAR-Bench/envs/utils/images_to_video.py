import cv2
import numpy as np
import os
import subprocess
import pickle
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import librosa.display
from scipy import signal
import soundfile as sf


def _align_audio_with_video(audio_data, audio_sample_rate, fps, n_frames):
    video_duration = n_frames / fps
    target_audio_samples = max(1, int(round(video_duration * audio_sample_rate)))
    audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.size == 0:
        return np.zeros(target_audio_samples, dtype=np.float32), video_duration
    if len(audio_data) == target_audio_samples:
        return audio_data.astype(np.float32, copy=False), video_duration
    resampled = signal.resample(audio_data, target_audio_samples).astype(np.float32, copy=False)
    return resampled, video_duration


def _merge_video_and_audio(video_path, audio_path, out_path):
    ffmpeg_merge = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", "-shortest", out_path,
        ]
    )
    if ffmpeg_merge.wait() != 0:
        raise IOError("Failed to merge video and audio")
    os.remove(video_path)
    os.remove(audio_path)


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = True,
                    audio_data: np.ndarray | None = None, audio_sample_rate: int = 16000) -> None:
    if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[3] not in (3, 4)):
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, C), with C equal to 3 or 4.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames, H, W, C = imgs.shape
    if C == 3:
        pixel_format = "rgb24" if is_rgb else "bgr24"
    else:
        pixel_format = "rgba"
    video_only_path = out_path if audio_data is None else out_path.replace(".mp4", "_temp_video.mp4")
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", pixel_format,
            "-video_size", f"{W}x{H}", "-framerate", str(fps),
            "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", video_only_path,
        ],
        stdin=subprocess.PIPE,
    )
    ffmpeg.stdin.write(imgs.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise IOError("Cannot open ffmpeg. Please check the output path and ensure ffmpeg is supported.")
    if audio_data is None:
        print(
            f"🎬 Video is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
        )
        return
    resized_audio, video_duration = _align_audio_with_video(audio_data, audio_sample_rate, fps, n_frames)
    temp_audio_path = out_path.replace(".mp4", "_temp_audio.wav")
    sf.write(temp_audio_path, resized_audio, audio_sample_rate)
    _merge_video_and_audio(video_only_path, temp_audio_path, out_path)
    print(
        f"🎬 Video with audio is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS (audio {video_duration:.2f}s)."
    )


def create_audio_waveform_image(audio_data, sample_rate, current_time, width=640, height=480, dpi=100, fig=None, ax=None):
    """
    Create an audio waveform visualization and optionally reuse an existing figure.
    """
    if fig is None or ax is None:
        fig_width = width / dpi
        fig_height = height / dpi
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)
        new_fig = True
    else:
        new_fig = False
        ax.clear()
    
    time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    ax.plot(time, audio_data, linewidth=0.5, color='blue')
    ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2, label='Current Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, time[-1] if len(time) > 0 else 1)
    ax.legend()
    
    if new_fig:
        plt.tight_layout()
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    width, height = fig.canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8)
    img = img.reshape(height, width, 4)
    img = img[:, :, :3]
    
    if new_fig:
        plt.close(fig)
    
    return img, fig, ax


def create_qpos_image(qpos_data, current_step, width=640, height=480, dpi=100, title="Joint Position", fig=None, ax=None):
    """
    Create a joint-position plot and optionally reuse an existing figure.
    """
    if fig is None or ax is None:
        fig_width = width / dpi
        fig_height = height / dpi
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)
        new_fig = True
    else:
        new_fig = False
        ax.clear()
    
    if len(qpos_data) > 0:
        n_steps, n_joints = qpos_data.shape
        steps = np.arange(n_steps)
        
        for joint_idx in range(n_joints):
            ax.plot(steps, qpos_data[:, joint_idx], linewidth=1.0, alpha=0.7, label=f'Joint {joint_idx+1}')

        ax.axvline(x=current_step, color='red', linestyle='--', linewidth=2, label='Current Time')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Position (rad)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(n_steps - 1, 1))
        
        if n_joints <= 7:
            ax.legend(fontsize='small', ncol=2)
    
    if new_fig:
        plt.tight_layout()
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    width, height = fig.canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8)
    img = img.reshape(height, width, 4)
    img = img[:, :, :3]
    
    if new_fig:
        plt.close(fig)
    
    return img, fig, ax


def combine_camera_views(head_img, left_img, right_img, audio_img, left_qpos_img, right_qpos_img):
    """
    Combine images into a 2x3 layout: cameras on top, plots on the bottom.
    """
    target_h, target_w = head_img.shape[:2]
    
    def resize_if_needed(img, target_h, target_w):
        if img.shape[:2] != (target_h, target_w):
            return cv2.resize(img, (target_w, target_h))
        return img
    
    left_img = resize_if_needed(left_img, target_h, target_w)
    right_img = resize_if_needed(right_img, target_h, target_w)
    audio_img = resize_if_needed(audio_img, target_h, target_w)
    left_qpos_img = resize_if_needed(left_qpos_img, target_h, target_w)
    right_qpos_img = resize_if_needed(right_qpos_img, target_h, target_w)
    
    def add_title(img, title):
        img_with_title = img.copy()
        cv2.putText(img_with_title, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img_with_title
    
    head_img = add_title(head_img, "Head Camera")
    left_img = add_title(left_img, "Left Camera")
    right_img = add_title(right_img, "Right Camera")
    audio_img = add_title(audio_img, "Audio Waveform")
    left_qpos_img = add_title(left_qpos_img, "Left Arm Joints")
    right_qpos_img = add_title(right_qpos_img, "Right Arm Joints")
    
    top_row = np.hstack([head_img, left_img, right_img])
    bottom_row = np.hstack([audio_img, left_qpos_img, right_qpos_img])
    combined = np.vstack([top_row, bottom_row])
    
    return combined


def combine_camera_audio_grid(head_img, left_img, right_img, audio_img):
    target_h, target_w = head_img.shape[:2]
    def resize(img):
        return cv2.resize(img, (target_w, target_h)) if img.shape[:2] != (target_h, target_w) else img
    def add_title(img, title):
        img_copy = img.copy()
        cv2.putText(img_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img_copy
    head_img = add_title(head_img, "Head Camera")
    left_img = add_title(resize(left_img), "Left Camera")
    right_img = add_title(resize(right_img), "Right Camera")
    audio_img = add_title(resize(audio_img), "Audio Waveform")
    top_row = np.hstack([head_img, left_img])
    bottom_row = np.hstack([right_img, audio_img])
    return np.vstack([top_row, bottom_row])


def multi_camera_audio_to_video(
    head_imgs: np.ndarray,
    left_imgs: np.ndarray,
    right_imgs: np.ndarray,
    audio_data: np.ndarray,
    audio_sample_rate: int,
    left_qpos_data: np.ndarray,
    right_qpos_data: np.ndarray,
    out_path: str,
    fps: float = 30.0,
    sim_timestep: float = 1/250,
    save_freq: int = 1,
    render_mode: str = "full",
) -> None:
    """Render a video that combines multiple camera views, audio, and joint traces."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    n_frames = len(head_imgs)
    if n_frames == 0:
        raise ValueError("No frames to process")
    
    actual_sim_duration = n_frames * save_freq * sim_timestep
    print(f"Simulation: {n_frames} frames, save_freq={save_freq}, timestep={sim_timestep}")
    print(f"Actual simulation duration: {actual_sim_duration:.2f}s")
    
    video_duration = n_frames / fps
    print(f"Video duration: {video_duration:.2f}s at {fps} FPS")
    
    audio_data = np.asarray(audio_data, dtype=np.float32)
    audio_duration_sec = (len(audio_data) / audio_sample_rate) if audio_data.size else 0
    print(f"Original audio: {len(audio_data)} samples ({audio_duration_sec:.2f}s) at {audio_sample_rate}Hz")
    
    target_audio_samples = int(video_duration * audio_sample_rate)

    if target_audio_samples <= 0:
        raise ValueError("Video duration is zero, cannot synchronize audio.")
    if audio_data.size == 0:
        print("Warning: Empty audio supplied, using silence.")
        resampled_audio = np.zeros(target_audio_samples, dtype=np.float32)
    elif len(audio_data) != target_audio_samples:
        speed_factor = len(audio_data) / target_audio_samples
        print(f"Audio speed adjustment: {speed_factor:.3f}x ({'speed up' if speed_factor > 1 else 'slow down'})")
        resampled_audio = signal.resample(audio_data, target_audio_samples).astype(np.float32, copy=False)
        print(f"Resampled audio: {len(resampled_audio)} samples ({len(resampled_audio)/audio_sample_rate:.2f}s)")
    else:
        resampled_audio = audio_data
        print("Audio length matches video, no resampling needed")
    
    temp_audio_path = out_path.replace('.mp4', '_temp_audio.wav')
    sf.write(temp_audio_path, resampled_audio, audio_sample_rate)

    print(f"Combining frames with render_mode='{render_mode}'...")
    combined_frames = []

    need_audio_chart = render_mode in ("audio", "full")
    need_qpos_charts = render_mode == "full"
    audio_fig = audio_ax = None
    left_qpos_fig = left_qpos_ax = None
    right_qpos_fig = right_qpos_ax = None

    for i in range(n_frames):
        if render_mode == "head_only":
            combined = head_imgs[i]
        elif render_mode == "cameras":
            head_img = head_imgs[i].copy()
            left_img = left_imgs[i].copy()
            right_img = right_imgs[i].copy()

            target_h, target_w = head_img.shape[:2]
            if left_img.shape[:2] != (target_h, target_w):
                left_img = cv2.resize(left_img, (target_w, target_h))
            if right_img.shape[:2] != (target_h, target_w):
                right_img = cv2.resize(right_img, (target_w, target_h))
            
            cv2.putText(head_img, "Head Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(left_img, "Left Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(right_img, "Right Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            combined = np.hstack([head_img, left_img, right_img])
        elif render_mode == "audio":
            current_time = video_duration * (i / n_frames) if n_frames > 0 else 0
            if need_audio_chart:
                audio_img, audio_fig, audio_ax = create_audio_waveform_image(
                    resampled_audio,
                    audio_sample_rate,
                    current_time,
                    width=head_imgs.shape[2],
                    height=head_imgs.shape[1],
                    fig=audio_fig,
                    ax=audio_ax,
                )
            combined = combine_camera_audio_grid(
                head_imgs[i],
                left_imgs[i],
                right_imgs[i],
                audio_img,
            )
        else:  # render_mode == "full"
            current_time = video_duration * (i / n_frames) if n_frames > 0 else 0
            if need_audio_chart:
                audio_img, audio_fig, audio_ax = create_audio_waveform_image(
                    resampled_audio,
                    audio_sample_rate,
                    current_time,
                    width=head_imgs.shape[2],
                    height=head_imgs.shape[1],
                    fig=audio_fig,
                    ax=audio_ax,
                )
            if need_qpos_charts:
                left_qpos_img, left_qpos_fig, left_qpos_ax = create_qpos_image(
                    left_qpos_data,
                    current_step=i,
                    width=head_imgs.shape[2],
                    height=head_imgs.shape[1],
                    title="Left Arm Joint Position",
                    fig=left_qpos_fig,
                    ax=left_qpos_ax,
                )
                right_qpos_img, right_qpos_fig, right_qpos_ax = create_qpos_image(
                    right_qpos_data,
                    current_step=i,
                    width=head_imgs.shape[2],
                    height=head_imgs.shape[1],
                    title="Right Arm Joint Position",
                    fig=right_qpos_fig,
                    ax=right_qpos_ax,
                )
            combined = combine_camera_views(
                head_imgs[i],
                left_imgs[i],
                right_imgs[i],
                audio_img,
                left_qpos_img,
                right_qpos_img,
            )
        
        combined_frames.append(combined)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{n_frames} frames", end="\r")
    
    print(f"\nProcessed all {n_frames} frames")
    
    if need_audio_chart and audio_fig is not None:
        plt.close(audio_fig)
    if need_qpos_charts and left_qpos_fig is not None:
        plt.close(left_qpos_fig)
    if need_qpos_charts and right_qpos_fig is not None:
        plt.close(right_qpos_fig)
    
    combined_frames = np.array(combined_frames)
    
    temp_video_path = out_path.replace('.mp4', '_temp_video.mp4')
    print("Creating video without audio...")
    
    H, W = combined_frames.shape[1], combined_frames.shape[2]
    ffmpeg_video = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            temp_video_path,
        ],
        stdin=subprocess.PIPE,
    )
    ffmpeg_video.stdin.write(combined_frames.tobytes())
    ffmpeg_video.stdin.close()
    
    if ffmpeg_video.wait() != 0:
        raise IOError("Failed to create temporary video")
    
    print("Merging video and audio...")
    ffmpeg_merge = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            temp_video_path,
            "-i",
            temp_audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-shortest",
            out_path,
        ]
    )
    
    if ffmpeg_merge.wait() != 0:
        raise IOError("Failed to merge video and audio")
    
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    
    print(
        f"🎬 Multi-camera video with audio saved to `{out_path}`, "
        f"containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
    )
