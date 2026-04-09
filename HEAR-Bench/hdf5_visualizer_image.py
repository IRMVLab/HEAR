import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import argparse
import os
from tqdm import tqdm
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor
import time
import wave
import tempfile

DEFAULT_AUDIO_SR = 16000

plt.switch_backend('Agg')

class FastVisualizer:
    def __init__(self, chunk_len, save_path, fps=30, width=1280, height=720, crf=23, episode_id=None):
        self.chunk_len = chunk_len
        self.save_path = save_path
        self.fps = fps
        self.pipe = None
        self.fig = None
        self.episode_id = episode_id
        self.total_w = width
        self.total_h = height
        self.crf = crf
        self.cam_h = int(height * 0.4)
        self.plot_h = height - self.cam_h

        if not shutil.which("ffmpeg"):
            raise RuntimeError("Error: 'ffmpeg' not found.")

        self._init_plot_area()

    def _init_plot_area(self):
        """Initialize the plot area only; camera frames are handled by OpenCV."""
        dpi = 100
        self.fig = plt.figure(figsize=(self.total_w / dpi, self.plot_h / dpi), dpi=dpi)
        plt.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.1)
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.25, 1.0], wspace=0.15)

        # --- 1. Audio ---
        self.ax_audio = self.fig.add_subplot(gs[0])
        self.line_audio, = self.ax_audio.plot([], [], color="#1f77b4", lw=1)
        self.vline_audio = self.ax_audio.axvline(0, color="red", linestyle="--", linewidth=1.5)
        self.ax_audio.set_title("Audio", fontsize=10, pad=5)
        self.ax_audio.set_xlabel("Time (s)")
        self.ax_audio.grid(True, alpha=0.3)

        # --- 2. Joints ---
        self.qpos_axes = []
        self.lines_qpos = []
        self.vlines_qpos = []
        
        gs_joints = gridspec.GridSpecFromSubplotSpec(7, 2, subplot_spec=gs[1], wspace=0.1, hspace=0.4)
        qpos_labels = [f"L_arm_{i}" for i in range(6)] + ["L_grp"] + \
                      [f"R_arm_{i}" for i in range(6)] + ["R_grp"]
        
        for idx in range(14):
            row, col = idx % 7, idx // 7
            if idx >= 7: 
                row = idx - 7
                col = 1
            else:
                row = idx
                col = 0

            ax = self.fig.add_subplot(gs_joints[row, col])
            ax.set_xlim(0, max(self.chunk_len - 1, 1))
            
            l_hist, = ax.plot([], [], color="#ff7f0e", lw=1.2)
            vline = ax.axvline(0, color="red", linestyle="--", linewidth=1)
            
            ax.text(0.01, 0.75, qpos_labels[idx], transform=ax.transAxes, fontsize=7, 
                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=0))
            
            ax.set_yticklabels([])
            ax.set_yticks([])
            if row != 6: ax.set_xticklabels([])
            
            self.qpos_axes.append(ax)
            self.lines_qpos.append(l_hist)
            self.vlines_qpos.append(vline)

    def update(self, step_idx, data_dict):
        vis_img = data_dict['camera_image']  # RGB
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        vis_img_resized = cv2.resize(vis_img_bgr, (self.total_w, self.cam_h))
        stage_label = str(data_dict.get('stage', 'default'))
        overlay_text = f"Ep {self.episode_id}  Stage {stage_label}  Step {step_idx}" if self.episode_id is not None else f"Stage {stage_label}  Step {step_idx}"
        cv2.putText(
            vis_img_resized,
            overlay_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
        # Update Audio
        audio_window = data_dict['audio_window']
        sr = data_dict['audio_sr']
        safe_sr = max(sr, 1)
        downsample_rate = max(1, len(audio_window) // 500) 
        audio_disp = audio_window[::downsample_rate]
        audio_time = np.linspace(0, len(audio_window)/safe_sr, len(audio_disp))
        
        self.line_audio.set_data(audio_time, audio_disp)
        self.ax_audio.set_xlim(0, max(audio_time[-1] if len(audio_time)>0 else 0.01, 0.01))
        limit = max(np.max(np.abs(audio_disp)) if len(audio_disp) > 0 else 0, 0.01)
        self.ax_audio.set_ylim(-limit*1.2, limit*1.2)
        self.vline_audio.set_xdata([audio_time[-1] if len(audio_time)>0 else 0])

        # Update Qpos
        qpos_window = data_dict['qpos_window'] 
        x_hist = np.arange(len(qpos_window))
        current_x = max(len(qpos_window) - 1, 0)

        for idx in range(14):
            if idx < qpos_window.shape[1]:
                self.lines_qpos[idx].set_data(x_hist, qpos_window[:, idx])
                self.vlines_qpos[idx].set_xdata([current_x])
                vals = qpos_window[:, idx]
                if len(vals) > 0:
                    min_v, max_v = np.min(vals), np.max(vals)
                    margin = 0.1 if abs(max_v - min_v) < 1e-6 else (max_v - min_v) * 0.1
                    self.qpos_axes[idx].set_ylim(min_v - margin, max_v + margin)
            else:
                self.lines_qpos[idx].set_data([], [])

        self.fig.canvas.draw()
        plot_img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        if plot_img.shape[0] != self.plot_h or plot_img.shape[1] != self.total_w:
            plot_img = cv2.resize(plot_img, (self.total_w, self.plot_h))

        final_frame = np.vstack((vis_img_resized, plot_img))

        if self.pipe is None:
            self._init_pipe(final_frame.shape[1], final_frame.shape[0])
        
        try:
            self.pipe.stdin.write(final_frame.tobytes())
        except BrokenPipeError:
            pass

    def _init_pipe(self, w, h):
        cmd = [
            'ffmpeg',
            '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', str(self.fps),
            '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-crf', str(self.crf),
            '-loglevel', 'error',
            self.save_path
        ]
        self.pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def close(self):
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.wait()
        plt.close(self.fig)

def decode_image(data_bytes):
    if isinstance(data_bytes, bytes):
        nparr = np.frombuffer(data_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None: return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.zeros((480, 640, 3), dtype=np.uint8)

def _prepare_joint_array(data, target_len, target_cols):
    arr = np.zeros((target_len, target_cols)) if data is None else np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] < target_len:
        arr = np.pad(arr, ((0, target_len - arr.shape[0]), (0, 0)), constant_values=0)
    elif arr.shape[0] > target_len:
        arr = arr[:target_len]
    if arr.shape[1] < target_cols:
        arr = np.pad(arr, ((0, 0), (0, target_cols - arr.shape[1])), constant_values=0)
    elif arr.shape[1] > target_cols:
        arr = arr[:, :target_cols]
    return arr

def get_data_at_step(f, idx, length, chunk_len, audio_ds=None, audio_sr_ds=None, stage_ds=None, default_sr=DEFAULT_AUDIO_SR):
    cam_names = ["left_camera", "head_camera", "right_camera"]
    imgs = []
    obs_grp = f['observation']
    for cam in cam_names:
        if cam in obs_grp and 'rgb' in obs_grp[cam]:
            imgs.append(decode_image(obs_grp[cam]['rgb'][idx]))
        else:
            imgs.append(np.zeros((480, 640, 3), dtype=np.uint8))
    vis_img = np.concatenate(imgs, axis=1)

    start_idx = idx
    end_idx = min(idx + chunk_len, length)

    qpos_all = None
    if 'joint_action' in f:
        ja = f['joint_action']
        left_arm = _prepare_joint_array(ja['left_arm'][()] if 'left_arm' in ja else None, length, 6)
        right_arm = _prepare_joint_array(ja['right_arm'][()] if 'right_arm' in ja else None, length, 6)
        left_gripper = _prepare_joint_array(ja['left_gripper'][()] if 'left_gripper' in ja else None, length, 1)
        right_gripper = _prepare_joint_array(ja['right_gripper'][()] if 'right_gripper' in ja else None, length, 1)
        qpos_all = np.hstack([left_arm, left_gripper, right_arm, right_gripper])
    elif 'qpos' in f:
        qpos_all = f['qpos'][()]
        if qpos_all.shape[1] == 14: pass 
        else: qpos_all = np.pad(qpos_all, ((0,0),(0, max(0, 14-qpos_all.shape[1]))))[:, :14]
    
    if qpos_all is None: qpos_all = np.zeros((length, 14))

    qpos_window = qpos_all[start_idx:end_idx]

    # Audio logic
    audio_window = np.zeros(1)
    audio_sr = default_sr
    if audio_ds is not None and idx < audio_ds.shape[0]:
        audio_window = np.asarray(audio_ds[idx]).reshape(-1)
        if audio_sr_ds is not None and idx < audio_sr_ds.shape[0]:
            audio_sr = max(int(audio_sr_ds[idx]), 1)
    stage_label = "default"
    if stage_ds is not None and idx < stage_ds.shape[0]:
        val = stage_ds[idx]
        if isinstance(val, bytes):
            stage_label = val.decode("utf-8", errors="ignore")
        else:
            stage_label = str(np.asarray(val)).strip()
    return {'camera_image': vis_img, 'qpos_window': qpos_window,
            'audio_window': audio_window, 'audio_sr': audio_sr, 'stage': stage_label}

def save_audio_chunk(audio_chunk, sample_rate, filepath):
    audio_arr = np.asarray(audio_chunk).reshape(-1)
    if audio_arr.size == 0:
        return
    if not np.issubdtype(audio_arr.dtype, np.floating):
        max_val = float(np.iinfo(audio_arr.dtype).max)
        if max_val == 0:
            return
        audio_arr = audio_arr.astype(np.float32) / max_val
    audio_arr = np.clip(audio_arr, -1.0, 1.0)
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

def _normalize_audio_array(audio_arr):
    arr = np.asarray(audio_arr).reshape(-1)
    if arr.size == 0:
        return None
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    else:
        info = np.iinfo(arr.dtype)
        scale = max(abs(info.min), info.max)
        if scale == 0:
            return None
        arr = arr.astype(np.float32) / float(scale)
    return np.clip(arr, -1.0, 1.0)

def _stretch_audio_to_duration(audio_arr, sample_rate, duration_sec):
    if audio_arr is None or sample_rate <= 0 or duration_sec <= 0:
        return None
    desired_samples = max(int(round(duration_sec * sample_rate)), 1)
    if desired_samples == audio_arr.size:
        return audio_arr
    src_pos = np.linspace(0.0, 1.0, audio_arr.size, endpoint=False)
    dst_pos = np.linspace(0.0, 1.0, desired_samples, endpoint=False)
    return np.interp(dst_pos, src_pos, audio_arr).astype(np.float32)

def attach_episode_audio_to_video(video_path, frame_count, fps, audio_arr, sample_rate):
    normalized = _normalize_audio_array(audio_arr)
    duration = frame_count / max(float(fps), 1e-6)
    stretched = _stretch_audio_to_duration(normalized, sample_rate, duration)
    if stretched is None:
        return
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp_video = f"{video_path}.tmp_with_audio.mp4"
    try:
        save_audio_chunk(stretched, sample_rate, tmp_wav)
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', tmp_wav,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            tmp_video
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(tmp_video, video_path)
    except subprocess.CalledProcessError as err:
        print(f"Failed to mux episode audio for {os.path.basename(video_path)}: {err}")
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

def _save_multiview_image(image_rgb, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, image_rgb)

def _save_multiview_images(image_rgb, frame_dir):
    h, w, _ = image_rgb.shape
    if w % 3 != 0:
        return
    single_w = w // 3
    cam0 = image_rgb[:, 0:single_w, :]
    cam1 = image_rgb[:, single_w:2*single_w, :]
    cam2 = image_rgb[:, 2*single_w:3*single_w, :]
    _save_multiview_image(cam0, os.path.join(frame_dir, "cam0.png"))
    _save_multiview_image(cam1, os.path.join(frame_dir, "cam1.png"))
    _save_multiview_image(cam2, os.path.join(frame_dir, "cam2.png"))

def _save_audio_wave_image(audio_chunk, sample_rate, filepath, out_width, out_height, y_limit=1.0):
    audio_arr = np.asarray(audio_chunk).reshape(-1)
    if audio_arr.size == 0:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    safe_sr = max(int(sample_rate), 1)
    downsample_rate = max(1, len(audio_arr) // 500)
    audio_disp = audio_arr[::downsample_rate]
    audio_time = np.linspace(0, len(audio_arr) / safe_sr, len(audio_disp))

    out_w = max(int(out_width), 1)
    out_h = max(int(out_height), 1)
    dpi = 100
    fig = plt.figure(figsize=(out_w / dpi, out_h / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.plot(audio_time, audio_disp, color="#1f77b4", lw=1)
    safe_limit = max(float(y_limit), 1e-6)
    ax.set_ylim(-safe_limit, safe_limit)
    ax.set_xlim(0.0, max(len(audio_arr) / safe_sr, 1e-6))
    ax.set_axis_off()

    fig.canvas.draw()
    img_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    if img_rgb.shape[1] != out_w or img_rgb.shape[0] != out_h:
        img_rgb = cv2.resize(img_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(filepath, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def process_single_file(args_tuple):
    """Worker function for a single HDF5 file."""
    hdf5_path, output_dir, chunk_len, fps, width, height, crf, dump_audio, audio_root, dump_frames, frame_root, dump_audio_plots = args_tuple
    
    filename = os.path.basename(hdf5_path)
    episode_id = filename.replace('.hdf5', '').replace('episode_', '')
    save_path = os.path.join(output_dir, f"vis_episode_{episode_id}.mp4")
    
    if os.path.exists(save_path):
        print(f"Skipping {filename}, already exists.")
        return

    episode_audio_dir = None
    if dump_audio and audio_root:
        episode_audio_dir = os.path.join(audio_root, f"episode_{episode_id}")
        os.makedirs(episode_audio_dir, exist_ok=True)

    episode_output_dir = None
    if (dump_frames or dump_audio_plots) and frame_root:
        episode_output_dir = os.path.join(frame_root, f"episode_{episode_id}")
        os.makedirs(episode_output_dir, exist_ok=True)

    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'observation' not in f: return
            length = f['observation']['head_camera']['rgb'].shape[0]
            audio_ds = f['audio'] if 'audio' in f else None
            audio_sr_ds = f['audio_status']['sample_rate'] if 'audio_status' in f and 'sample_rate' in f['audio_status'] else None
            stage_ds = f['stage'] if 'stage' in f else None
            episode_sr = DEFAULT_AUDIO_SR
            if audio_sr_ds is not None:
                try:
                    episode_sr = max(int(np.asarray(audio_sr_ds[0]).item()), 1)
                except Exception:
                    sr_arr = np.asarray(audio_sr_ds).reshape(-1)
                    if sr_arr.size:
                        episode_sr = max(int(sr_arr[0]), 1)
            full_episode_audio = None
            full_episode_sr = episode_sr
            if 'full_episode_audio' in f:
                full_episode_audio = f['full_episode_audio']

            if full_episode_audio is not None and dump_audio and audio_root:
                full_audio_path = os.path.join(output_dir, f"full_episode_{episode_id}.wav")
                save_audio_chunk(full_episode_audio, full_episode_sr, full_audio_path)
            visualizer = FastVisualizer(chunk_len, save_path, fps=fps,
                                        width=width, height=height, crf=crf, episode_id=episode_id)

            iterator = range(length)
            if torch_rank == 0:
                iterator = tqdm(range(length), desc=f"Ep {episode_id}", leave=False)
                
            for i in iterator:
                data = get_data_at_step(f, i, length, chunk_len,
                                        audio_ds=audio_ds, audio_sr_ds=audio_sr_ds, stage_ds=stage_ds, default_sr=episode_sr)
                if episode_audio_dir and data['audio_window'].size:
                    chunk_path = os.path.join(episode_audio_dir, f"{i:06d}.wav")
                    save_audio_chunk(data['audio_window'], data['audio_sr'], chunk_path)
                if episode_output_dir:
                    frame_dir = os.path.join(episode_output_dir, f"{i:06d}")
                    os.makedirs(frame_dir, exist_ok=True)
                    if dump_frames:
                        _save_multiview_images(data['camera_image'], frame_dir)
                    if dump_audio_plots and data['audio_window'].size:
                        h, w, _ = data['camera_image'].shape
                        audio_w = w
                        audio_h = max(h // 3, 1)
                        _save_audio_wave_image(
                            data['audio_window'],
                            data['audio_sr'],
                            os.path.join(frame_dir, "audio.png"),
                            audio_w,
                            audio_h,
                            y_limit=1.0
                        )
                        _save_audio_wave_image(
                            data['audio_window'],
                            data['audio_sr'],
                            os.path.join(frame_dir, "audio_pm0p1.png"),
                            audio_w,
                            audio_h,
                            y_limit=0.1
                        )
                visualizer.update(i, data)
            
            visualizer.close()
            if full_episode_audio is not None:
                attach_episode_audio_to_video(save_path, length, fps, full_episode_audio, full_episode_sr)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

torch_rank = 0 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default="data/touch_plate_metal/demo_clean_audio/data")
    parser.add_argument('--output', '-o', type=str, default="visualizations_replay")
    parser.add_argument('--chunk_len', type=int, default=32)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--width', type=int, default=640, help="Output video width.")
    parser.add_argument('--height', type=int, default=360, help="Output video height.")
    parser.add_argument('--crf', type=int, default=28, help="FFmpeg CRF (lower = higher quality).")
    parser.add_argument('--workers', '-w', type=int, default=10, help="Number of parallel processes")
    parser.add_argument('--dump-audio', action='store_true', help="Dump each frame's audio window as WAV.", default=True)
    parser.add_argument('--audio-dir', type=str, default=None, help="Destination for dumped audio chunks.")
    parser.add_argument('--dump-frame-images', action='store_true', help="Dump per-frame multiview images.", default=True)
    parser.add_argument('--frame-image-dir', type=str, default=None, help="Destination for dumped frame images.")
    parser.add_argument('--dump-audio-plots', action='store_true', help="Dump per-frame audio waveform images.", default=True)
    parser.add_argument('--audio-plot-dir', type=str, default=None, help="Destination for dumped audio plots.")
    parser.add_argument('--merge-all', action='store_true', help="After all episodes exported, merge into one video.", default=True)
    parser.add_argument('--merged-name', type=str, default="merged_episodes.mp4", help="Filename for merged video inside output dir.")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    files = []
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.hdf5')])
    
    print(f"Found {len(files)} files. Using {args.workers} workers.")
    
    audio_dir = args.audio_dir or os.path.join(args.output, "audio_chunks")
    if not args.dump_audio:
        audio_dir = None
    elif audio_dir:
        os.makedirs(audio_dir, exist_ok=True)

    frame_dir = args.frame_image_dir or args.audio_plot_dir or os.path.join(args.output, "frame_images")
    if not (args.dump_frame_images or args.dump_audio_plots):
        frame_dir = None
    elif frame_dir:
        os.makedirs(frame_dir, exist_ok=True)
    
    tasks = [
        (f, args.output, args.chunk_len, args.fps, args.width, args.height, args.crf,
         args.dump_audio, audio_dir, args.dump_frame_images, frame_dir, args.dump_audio_plots)
        for f in files
    ]
    
    if args.workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(process_single_file, tasks), total=len(tasks), desc="Total Progress"))
    else:
        for task in tasks:
            process_single_file(task)

    if args.merge_all:
        vid_files = sorted([os.path.join(args.output, f) for f in os.listdir(args.output)
                            if f.startswith("vis_episode_") and f.endswith(".mp4")])
        vid_files = [os.path.abspath(v) for v in vid_files if os.path.exists(v)]
        if not vid_files:
            print("No episode videos found to merge.")
            return
        list_file = os.path.join(args.output, "concat_list.txt")
        with open(list_file, "w", encoding="utf-8") as lf:
            for vf in vid_files:
                lf.write(f"file '{vf}'\n")
        merged_path = os.path.abspath(os.path.join(args.output, args.merged_name))
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', os.path.abspath(list_file), '-c', 'copy', merged_path]
        try:
            subprocess.run(cmd, check=True)
            print(f"Merged {len(vid_files)} videos -> {merged_path}")
        except subprocess.CalledProcessError as err:
            print(f"Failed to merge videos: {err}")
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)

if __name__ == "__main__":
    main()
