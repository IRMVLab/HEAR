import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
import threading
import queue
import cv2
import os
import shutil

class RealTimeVisualizer:
    def __init__(self, chunk_len, save_dir="visualizations"):
        self.chunk_len = chunk_len
        self.save_dir_root = save_dir
        self.save_dir = None
        self.fig = plt.figure(figsize=(24, 12)) 
        plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)

        outer = gridspec.GridSpec(2, 1, height_ratios=[1, 2.2], figure=self.fig, hspace=0.25)

        top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.05)
        self.ax_input = self.fig.add_subplot(top[0, 0])
        self.ax_current = self.fig.add_subplot(top[0, 1])
        self.im_input = None
        self.im_current = None

        bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], 
                                                  wspace=0.15, width_ratios=[0.4, 1.3, 1.3])

        audio_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=bottom[0, 0], hspace=0.5)
        
        # --- Input Audio (Top) ---
        self.ax_audio_input = self.fig.add_subplot(audio_gs[0, 0])
        self.line_audio_input, = self.ax_audio_input.plot([], [], color="#1f77b4")
        self.vline_audio_input = self.ax_audio_input.axvline(0, color="red", linestyle="--", linewidth=1.5)
        self.vline_exec_audio_input = self.ax_audio_input.axvline(0, color="blue", linestyle="-", linewidth=1.2)
        self.ax_audio_input.set_title("Model Input Audio", fontsize=12, pad=5)
        self.ax_audio_input.yaxis.set_major_locator(MaxNLocator(nbins=3))
        self.ax_audio_input.tick_params(axis='y', labelsize=7)
        
        # --- Real-time Audio (Bottom) ---
        self.ax_audio_curr = self.fig.add_subplot(audio_gs[1, 0])
        self.line_audio_curr, = self.ax_audio_curr.plot([], [], color="#ff7f0e")
        self.vline_audio_curr = self.ax_audio_curr.axvline(0, color="red", linestyle="--", linewidth=1.5)
        self.vline_exec_audio_curr = self.ax_audio_curr.axvline(0, color="blue", linestyle="-", linewidth=1.2)
        self.ax_audio_curr.set_title("Real-time Observation Audio", fontsize=12, pad=5)
        self.ax_audio_curr.yaxis.set_major_locator(MaxNLocator(nbins=3))
        self.ax_audio_curr.tick_params(axis='y', labelsize=7)

        self.qpos_axes = []
        self.lines_action = []
        self.lines_history = []
        self.vlines_qpos = []
        self.vlines_exec_qpos = []
        
        qpos_grid = gridspec.GridSpecFromSubplotSpec(7, 2, subplot_spec=bottom[0, 1], wspace=0.35, hspace=0.15)
        qpos_labels = [f"L_arm_{i}" for i in range(6)] + ["L_gripper"] + \
                      [f"R_arm_{i}" for i in range(6)] + ["R_gripper"]
        
        for idx in range(14):
            if idx < 7:
                row, col = idx, 0
                title_text = "Joint Positions (Left)" if idx == 0 else None
            else:
                row, col = idx - 7, 1
                title_text = "Joint Positions (Right)" if idx == 7 else None

            ax = self.fig.add_subplot(qpos_grid[row, col])
            ax.set_xlim(0, max(chunk_len - 1, 1))
            
            if title_text:
                ax.set_title(title_text, fontsize=12, pad=5, color="#333333")

            l_act, = ax.plot([], [], color="#1f77b4", linestyle="--", linewidth=1, label="action")
            l_hist, = ax.plot([], [], color="#ff7f0e", linestyle="-", linewidth=1, label="history")
            vline = ax.axvline(0, color="red", linestyle="--", linewidth=1)
            vline_exec = ax.axvline(0, color="blue", linestyle="-", linewidth=1)
            
            label = qpos_labels[idx]
            ax.text(0.02, 0.75, label, transform=ax.transAxes, fontsize=8, fontweight='bold',
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5))
            
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None)) 
            ax.tick_params(axis='y', labelsize=7, pad=1)
            
            if row != 6: 
                ax.set_xticklabels([])
            
            self.qpos_axes.append(ax)
            self.lines_action.append(l_act)
            self.lines_history.append(l_hist)
            self.vlines_qpos.append(vline)
            self.vlines_exec_qpos.append(vline_exec)

        self.endpose_axes = []
        self.lines_endpose = []
        self.vlines_endpose = []
        self.vlines_exec_endpose = []
        
        endpose_grid = gridspec.GridSpecFromSubplotSpec(7, 2, subplot_spec=bottom[0, 2], wspace=0.35, hspace=0.15)
        endpose_labels = [f"L_{axis}" for axis in ["x", "y", "z", "qw", "qx", "qy", "qz"]] + \
                         [f"R_{axis}" for axis in ["x", "y", "z", "qw", "qx", "qy", "qz"]]

        for idx in range(14):
            if idx < 7:
                row, col = idx, 0
                title_text = "EE Pose (Left)" if idx == 0 else None
            else:
                row, col = idx - 7, 1
                title_text = "EE Pose (Right)" if idx == 7 else None

            ax = self.fig.add_subplot(endpose_grid[row, col])
            ax.set_xlim(0, max(chunk_len - 1, 1))

            if title_text:
                ax.set_title(title_text, fontsize=12, pad=5, color="#333333")

            l_pose, = ax.plot([], [], color="#2ca02c", linewidth=1)
            vline = ax.axvline(0, color="red", linestyle="--", linewidth=1)
            vline_exec = ax.axvline(0, color="blue", linestyle="-", linewidth=1)
            
            label = endpose_labels[idx]
            ax.text(0.02, 0.75, label, transform=ax.transAxes, fontsize=8, fontweight='bold',
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5))
            
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis='y', labelsize=7, pad=1)

            if row != 6: 
                ax.set_xticklabels([])
            
            self.endpose_axes.append(ax)
            self.lines_endpose.append(l_pose)
            self.vlines_endpose.append(vline)
            self.vlines_exec_endpose.append(vline_exec)

        self.save_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.worker_thread.start()

    def _init_cam_plot(self, ax, img, title):
        im = ax.imshow(img, aspect='auto')
        ax.set_title(title, fontsize=16, pad=10)
        ax.axis("off")
        h, w, _ = img.shape
        seg_w = w / 3.0
        for idx, label in enumerate(["left", "head", "right"]):
            ax.text(idx * seg_w + 20, 40, label, color="white", fontsize=14, weight="bold",
                    bbox=dict(facecolor="black", alpha=0.5, pad=3, edgecolor="none"))
        return im

    def _update_single_audio(self, ax, line, vline, vline_exec, audio_data, sr, current_id, exec_step):
        """Update one audio subplot, including the execution marker."""
        audio_wave = audio_data if audio_data is not None else np.zeros(1)
        audio_time = np.arange(len(audio_wave)) / max(sr, 1)
        
        if len(line.get_xdata()) != len(audio_time):
             line.set_data(audio_time, audio_wave)
             ax.set_xlim(audio_time[0], audio_time[-1] if len(audio_time)>1 else 1)
        else:
             line.set_ydata(audio_wave)

        limit = max(np.max(np.abs(audio_wave)), 0.01)
        ax.set_ylim(-limit*1.2, limit*1.2)

        current_audio_t = (current_id / max(self.chunk_len - 1, 1)) * audio_time[-1] if len(audio_time) > 0 else 0
        vline.set_xdata([current_audio_t])
        exec_audio_t = (exec_step / max(self.chunk_len - 1, 1)) * audio_time[-1] if len(audio_time) > 0 else 0
        vline_exec.set_xdata([exec_audio_t])

    def update(self, input_observation, observation, id, actions, qpos_history, endpose_history, episode_step, vis_idx, episode_id, executing_chunk_len=None):
        if self.save_dir is None or not self.save_dir.endswith(f"episode_{episode_id}"):
            self.save_dir = os.path.join(self.save_dir_root, f"episode_{episode_id}")
            os.makedirs(self.save_dir, exist_ok=True)

        # 1. Update Cameras
        input_imgs = [input_observation["observation"][k]["rgb"] for k in ["left_camera", "head_camera", "right_camera"]]
        input_vis_img = np.concatenate(input_imgs, axis=1)
        curr_imgs = [observation["observation"][k]["rgb"] for k in ["left_camera", "head_camera", "right_camera"]]
        vis_img = np.concatenate(curr_imgs, axis=1)

        if self.im_input is None:
            self.im_input = self._init_cam_plot(self.ax_input, input_vis_img, "Model Input Cameras")
            self.im_current = self._init_cam_plot(self.ax_current, vis_img, "Real-time Cameras")
        else:
            self.im_input.set_data(input_vis_img)
            self.im_current.set_data(vis_img)

        max_step = self.chunk_len - 1
        exec_step = max_step if (executing_chunk_len is None or executing_chunk_len > max_step) else executing_chunk_len

        sr = input_observation.get("audio_status", {}).get("sample_rate", 
             observation.get("audio_status", {}).get("sample_rate", 16000))

        input_audio_data = input_observation.get("audio", None)
        self._update_single_audio(self.ax_audio_input, self.line_audio_input, self.vline_audio_input, 
                                  self.vline_exec_audio_input, input_audio_data, sr, id, exec_step)

        current_audio_data = observation.get("audio", None)
        self._update_single_audio(self.ax_audio_curr, self.line_audio_curr, self.vline_audio_curr, 
                                  self.vline_exec_audio_curr, current_audio_data, sr, id, exec_step)

        # 3. Update Qpos & Actions
        actions_arr = np.asarray(actions)
        qpos_hist = np.asarray(qpos_history)
        x_actions = np.arange(actions_arr.shape[0])
        x_history = np.arange(len(qpos_hist))

        for idx in range(14):
            self.lines_action[idx].set_data(x_actions, actions_arr[:, idx])
            self.lines_history[idx].set_data(x_history, qpos_hist[:, idx])
            self.vlines_qpos[idx].set_xdata([id])
            self.vlines_exec_qpos[idx].set_xdata([exec_step])
            
            if len(qpos_hist) > 0:
                vals = []
                if len(actions_arr) > idx: vals.append(actions_arr[:, idx])
                if len(qpos_hist) > 0: vals.append(qpos_hist[:, idx])
                
                if vals:
                    all_vals = np.concatenate(vals)
                    min_v, max_v = np.min(all_vals), np.max(all_vals)
                    if abs(max_v - min_v) < 1e-6:
                        margin = 0.1
                    else:
                        margin = (max_v - min_v) * 0.15
                    self.qpos_axes[idx].set_ylim(min_v - margin, max_v + margin)

        # 4. Update Endpose
        endpose_arr = np.asarray(endpose_history)
        endpose_hist = endpose_arr.reshape(endpose_arr.shape[0], -1)
        
        for idx in range(14):
            self.lines_endpose[idx].set_data(x_history, endpose_hist[:, idx])
            self.vlines_endpose[idx].set_xdata([id])
            self.vlines_exec_endpose[idx].set_xdata([exec_step])
            
            if len(endpose_hist) > 0:
                min_v, max_v = np.min(endpose_hist[:, idx]), np.max(endpose_hist[:, idx])
                if abs(max_v - min_v) < 1e-6:
                    margin = 0.1
                else:
                    margin = (max_v - min_v) * 0.15
                self.endpose_axes[idx].set_ylim(min_v - margin, max_v + margin)

        self.fig.suptitle(f"Visualization @ step {episode_step} (id={id})", fontsize=20, y=0.98)
        
        self.fig.canvas.draw() 
        img_rgba = np.asarray(self.fig.canvas.buffer_rgba())
        img_rgb = img_rgba[:, :, :3]
        
        save_path = os.path.join(self.save_dir, f"step_{vis_idx:05d}.png")
        self.save_queue.put((save_path, img_rgb.copy()))

    def _save_worker(self):
        while True:
            save_path, img_rgb = self.save_queue.get()
            if save_path is None: break
            try:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, img_bgr)
            except Exception as e:
                print(f"Error saving image: {e}")
            finally:
                self.save_queue.task_done()
    
    def close(self):
        plt.close(self.fig)

    def delete_episode_dir(self, episode_id=None):
        """
        Delete one episode directory, or all episode directories if episode_id is None.
        """
        if episode_id is not None:
            dir_path = os.path.join(self.save_dir_root, f"episode_{episode_id}")
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        else:
            for name in os.listdir(self.save_dir_root):
                if name.startswith("episode_"):
                    dir_path = os.path.join(self.save_dir_root, name)
                    if os.path.isdir(dir_path):
                        shutil.rmtree(dir_path)

    def episode_to_video(self, episode_id, fps=20):
        """Encode one episode directory into a video, then remove the PNG frames."""
        dir_path = os.path.join(self.save_dir_root, f"episode_{episode_id}")
        if not os.path.exists(dir_path):
            print(f"Episode dir {dir_path} does not exist.")
            return

        img_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".png")])
        if not img_files:
            print(f"No images found in {dir_path}.")
            return

        video_path = os.path.join(self.save_dir_root, f"episode_{episode_id}.mp4")

        cmd = (
            f"ffmpeg -y -framerate {fps} -i {dir_path}/step_%05d.png "
            f"-c:v libx264 -pix_fmt yuv420p {video_path}"
        )
        print(f"Running: {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            print("ffmpeg failed.")
            return

        for img_name in img_files:
            img_path = os.path.join(dir_path, img_name)
            os.remove(img_path)

        print(f"Video saved to {video_path}")
        self.save_dir = None
