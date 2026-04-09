from copy import deepcopy
import os

import numpy as np

from ._base_task import Base_Task
from .utils import *


class pour_water_audio_half(Base_Task):
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.pour_audio_path = os.path.join("assets", "audios", "pour_water_half.mp3")
        self.audio_started = False
        self.audio_completed = False
        self.audio_start_step = -1
        self.audio_end_step = -1
        self.audio_gated_motion = False

    def load_actors(self):
        self.bottle = rand_create_actor(
            self,
            xlim=[-0.3, -0.2],
            ylim=[0.03, 0.15],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.66, 0.66, -0.25, -0.25],
            convex=True,
            model_id=13,
        )

        self.can = rand_create_actor(
            scene=self,
            modelname="071_can",
            model_id=3,
            xlim=[-0.05, 0.05],
            ylim=[-0.15, -0.05],
            qpos=[0.707225, 0.706849, -0.0100455, -0.00982061],
            convex=True,
        )
        self.can.set_mass(0.01)
        self.add_prohibit_area(self.can, padding=0.1)

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.add_prohibit_area(self.bottle, padding=0.1)

        can_pose = deepcopy(self.can.get_pose())
        xy = can_pose.p[:2]
        self.left_target_pose = [xy[0] - 0.12, xy[1] - 0.04, 1, -0.5, 0.5, 0.5, -0.5]
        self.left_final_pose = [0.00, -0.3, 1.0, 0, 1, 0, 0]

    def _wait_and_update(self, duration, save_freq=None):
        wait_steps = int(duration / self.scene.get_timestep())
        save_freq = self.save_freq if save_freq is None else save_freq
        for i in range(wait_steps):
            self.scene.step()
            if self.collect_audio:
                self.audio.update()
            if self.render_freq:
                self._update_render()
                self.viewer.render()
            if save_freq is not None and i % save_freq == 0:
                self._update_render()
                self._take_picture()
        return wait_steps

    def play_half(self):
        bottle_arm_tag = ArmTag("left")

        self.set_stage_label("grasp")
        self.move(
            self.grasp_actor(self.bottle, arm_tag=bottle_arm_tag, pre_grasp_dis=0.08),
        )

        self.set_stage_label("lift")
        self.move(
            self.move_by_displacement(arm_tag=bottle_arm_tag, z=0.1),
        )

        self.set_stage_label("move_to_pour_pose")
        self.set_stage_label("pour")
        self.move(
            self.place_actor(
                self.bottle,
                target_pose=self.left_target_pose,
                arm_tag=bottle_arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
            ),
        )

    def play_once(self):
        bottle_arm_tag = ArmTag("left")

        self.set_stage_label("grasp")
        self.move(
            self.grasp_actor(self.bottle, arm_tag=bottle_arm_tag, pre_grasp_dis=0.08),
        )

        self.set_stage_label("lift")
        self.move(
            self.move_by_displacement(arm_tag=bottle_arm_tag, z=0.1),
        )

        self.set_stage_label("move_to_pour_pose")
        self._play_pour_audio(delay=1.7)

        self.set_stage_label("pour")
        self.move(
            self.place_actor(
                self.bottle,
                target_pose=self.left_target_pose,
                arm_tag=bottle_arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
            ),
        )

        self.set_stage_label("padding")
        self._wait_and_update(2.05, save_freq=self.save_freq)

        self.set_stage_label("move_to_final_pose")
        self.move(
            self.place_actor(
                self.bottle,
                target_pose=self.left_final_pose,
                arm_tag=bottle_arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
            ),
        )
        self.audio_gated_motion = True
        self.info["info"] = {
            "{A}": "001_bottle/base13",
            "{B}": "071_can/base3",
            "audio_started": self.audio_started,
            "audio_completed": self.audio_completed,
            "audio_start_step": self.audio_start_step,
            "audio_end_step": self.audio_end_step,
        }
        return self.info

    def _play_pour_audio(self, delay=0.0):
        if not getattr(self, "collect_audio", False) or getattr(self, "audio", None) is None:
            self.audio_completed = True
            return
        if not os.path.exists(self.pour_audio_path):
            print(f"Warning: missing audio file {self.pour_audio_path}")
            self.audio_completed = True
            return
        success = self.audio.start_playing(
            self.pour_audio_path,
            loop_audio=False,
            randomize_start=False,
            delay=delay,
        )
        if success:
            self.audio_started = True
            self.audio_start_step = self.audio.total_steps
        else:
            self.audio_completed = True

    def _wait_for_audio_completion(self):
        if self.audio_completed:
            return
        if not getattr(self, "collect_audio", False) or getattr(self, "audio", None) is None:
            self.audio_completed = True
            return
        if not self.audio_started:
            self.audio_completed = True
            return
        while self.audio.is_playing:
            self.scene.step()
            self.audio.update()
            if getattr(self, "render_freq", 0):
                self._update_render()
                viewer = getattr(self, "viewer", None)
                if viewer:
                    viewer.render()
        self.audio_completed = True
        self.audio_end_step = self.audio.total_steps

    def _handle_step_audio(self):
        if getattr(self, "audio", None) is None or not getattr(self, "collect_audio", False):
            return
        if getattr(self, "audio_started", False) or getattr(self, "audio_completed", False):
            return
        if not hasattr(self, "bottle") or not hasattr(self, "left_target_pose"):
            return
        target_pose = [-0.1827675, -0.10443872, 0.9990898, 0.62350756, 0.33228886, 0.33520183, -0.6232675]
        if self._is_actor_near_pose(self.bottle, target_pose):
            self._play_pour_audio(delay=0.0)

    def _is_actor_near_pose(self, actor, target_pose=None, angle_eps=np.deg2rad(10)):
        def _extract_quat(pose_like):
            if hasattr(pose_like, "q"):
                q_raw = np.asarray(pose_like.q, dtype=float)
                return np.asarray([q_raw[-1], *q_raw[:3]], dtype=float)
            if isinstance(pose_like, (list, tuple, np.ndarray)):
                arr = np.asarray(pose_like, dtype=float)
                if arr.size >= 7:
                    return arr[3:7]
                if arr.size == 4:
                    return arr
            return None

        if actor is None or target_pose is None:
            return False
        target_q = _extract_quat(target_pose)
        actor_pose = deepcopy(actor.get_pose())
        actor_pose = list(actor_pose.p) + list(actor_pose.q)
        actor_q = _extract_quat(actor_pose)
        if target_q is None or actor_q is None:
            return False
        target_q = target_q / np.linalg.norm(target_q)
        actor_q = actor_q / np.linalg.norm(actor_q)
        ang = 2.0 * np.arccos(np.clip(np.abs(np.dot(actor_q, target_q)), -1.0, 1.0))
        return ang < angle_eps

    def check_success(self):
        bottle_target = self.left_final_pose[:2]
        eps = 0.1

        bottle_pose = self.bottle.get_functional_point(0)
        if bottle_pose[2] < 0.78:
            self.actor_pose = False
        return (
            abs(bottle_pose[0] - bottle_target[0]) < eps
            and abs(bottle_pose[1] - bottle_target[1]) < eps
            and bottle_pose[2] > 0.89
        )
