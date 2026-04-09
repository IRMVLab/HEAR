from ._base_task import Base_Task
from .utils import *
import sapien
import math
import os
import numpy as np


class show_bottle_yes(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.scene_side = self._init_scene_side("right")
        self.yes_audio_path = os.path.join("assets", "audios", "yes!.mp3")
        self.question_audio_path = os.path.join("assets", "audios", "yes?.mp3")
        self.yes_audio_started = False
        self.question_audio_started = False
        self.yes_audio_step = -1
        self.question_audio_step = -1
        self.audio_delay_time = 0.0
        self.left_grasp_arm_pose = [-0.16047078371047974, -0.05700467526912689, 0.8972667455673218]

    def _init_scene_side(self, side):
        valid = {"left", "right"}
        if side is None or side == "random":
            return np.random.choice(list(valid))
        side = str(side).lower()
        if side not in valid:
            raise ValueError("scene_side must be 'left' or 'right'")
        return side

    def _play_yes_audio(self, audio_path, started_attr, step_attr, delay=0.0):
        file_exists = os.path.exists(audio_path)
        if not file_exists:
            print(f"Warning: missing audio file {audio_path}")
        if (
            not getattr(self, "collect_audio", False)
            or getattr(self, "audio", None) is None
            or not file_exists
        ):
            return False
        success = self.audio.start_playing(
            audio_path,
            loop_audio=False,
            randomize_start=False,
            delay=delay,
        )
        if success:
            setattr(self, started_attr, True)
            setattr(self, step_attr, self.audio.total_steps)
        return success

    def load_actors(self):

        self.bottle_1 = rand_create_actor(
            self,
            xlim=[-0.15, -0.15],
            ylim=[-0.18, -0.18],
            zlim=[0.752],
            rotate_rand=False,
            qpos=[1.0, 0.0, 0.0, 0.0],
            modelname="001_bottle",
            convex=True,
            model_id=13,
        )
        self.bottle_2 = rand_create_actor(
            self,
            xlim=[0.15, 0.15],
            ylim=[-0.18, -0.18],
            zlim=[0.752],
            rotate_rand=False,
            # qpos=[0.707, 0.0, 0.0, -0.707],
            qpos=[1.0, 0.0, 0.0, 0.0],
            modelname="001_bottle",
            convex=True,
            # rotate_lim=(0, 0, 0.4),
            model_id=16,
        )
        self.delay(4)
        self.add_prohibit_area(self.bottle_1, padding=0.15)
        self.add_prohibit_area(self.bottle_2, padding=0.15)
        self.left_target_pose = [-0.20, 0.2, 1.0, 0, 1, 0, 0]
        self.right_target_pose = [0.20, 0.2, 1.0, 0, 1, 0, 0]

    def _wait_and_update(self, duration, save_freq=None):
        wait_steps = int(duration / self.scene.get_timestep())
        save_freq = self.save_freq if save_freq is None else save_freq
        for i in range(wait_steps):
            self.scene.step()
            if getattr(self, "collect_audio", False) and getattr(self, "audio", None):
                self.audio.update()
            if self.render_freq:
                self._update_render()
                self.viewer.render()
            if save_freq is not None and i % save_freq == 0:
                self._update_render()
                self._take_picture()
        return wait_steps

    def play_half(self):
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        if self.scene_side == "left":
            self.move(self.grasp_actor(self.bottle_1, arm_tag=left_arm_tag, pre_grasp_dis=0.1))
            # self.move(self.move_by_displacement(arm_tag=left_arm_tag, z=0.23, move_axis="arm"))

        else:
            self.move(self.grasp_actor(self.bottle_1, arm_tag=left_arm_tag, pre_grasp_dis=0.1))
            self.move(self.move_by_displacement(arm_tag=left_arm_tag, z=0.23, move_axis="arm"))
            # self.move(self.grasp_actor(self.bottle_2, arm_tag=right_arm_tag, pre_grasp_dis=0.1))

        self.delay(3)

    def play_once(self):
        
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        random_delay_time = float(np.random.uniform(3.5, 4.5))
        # random_delay_time = 4.5

        if self.scene_side == "left":
            self.set_stage_label("grasp_left")
            self._play_yes_audio(self.yes_audio_path, "yes_audio_started", "yes_audio_step", delay=random_delay_time)
            self.move(self.grasp_actor(self.bottle_1, arm_tag=left_arm_tag, pre_grasp_dis=0.1))
            self.move(self.move_by_displacement(arm_tag=left_arm_tag, z=0.23, move_axis="arm"))
            self.move(
                self.place_actor(
                    self.bottle_1,
                    target_pose=self.left_target_pose,
                    arm_tag=left_arm_tag,
                    functional_point_id=0,
                    pre_dis=0.0,
                    is_open=False,
                ))
        else:
            self.set_stage_label("grasp_left")
            self._play_yes_audio(self.question_audio_path, "question_audio_started", "question_audio_step", delay=random_delay_time)
            self.move(self.grasp_actor(self.bottle_1, arm_tag=left_arm_tag, pre_grasp_dis=0.1))
            self.move(self.move_by_displacement(arm_tag=left_arm_tag, z=0.23, move_axis="arm"))

            self.set_stage_label("grasp_right")
            self.move(self.grasp_actor(self.bottle_2, arm_tag=right_arm_tag, pre_grasp_dis=0.1))
            self.move(self.move_by_displacement(arm_tag=right_arm_tag, z=0.23, move_axis="arm"))
            self.move(
                self.place_actor(
                    self.bottle_2,
                    target_pose=self.right_target_pose,
                    arm_tag=right_arm_tag,
                    functional_point_id=0,
                    pre_dis=0.0,
                    is_open=False,
                ))

        self.info["info"] = {
            "{A}": f"001_bottle/base13",
            "{a}": str(left_arm_tag),
            "{B}": f"001_bottle/base16",
            "{b}": str(right_arm_tag),
            "{side}": self.scene_side,
        }
        self.info["audio"] = {
            "scene_side": self.scene_side,
            "yes_audio_started": self.yes_audio_started,
            "yes_audio_step": self.yes_audio_step,
            "question_audio_started": self.question_audio_started,
            "question_audio_step": self.question_audio_step,
        }
        return self.info

    def check_success(self):
        target_pose = self.left_target_pose if self.scene_side == "left" else self.right_target_pose
        bottle_pose = (
            self.bottle_1.get_functional_point(0)
            if self.scene_side == "left"
            else self.bottle_2.get_functional_point(0)
        )
        target_pos = np.array(target_pose[:3])
        bottle_pos = np.array(bottle_pose[:3])
        return np.linalg.norm(bottle_pos - target_pos) <= 0.05

    def _handle_step_audio(self):
        if getattr(self, "audio", None) is None or not getattr(self, "collect_audio", False):
            return
        if getattr(self, "yes_audio_started", False) or getattr(self, "question_audio_started", False):
            return

        actor = getattr(self, "bottle_1", None)
        try:
            actor_fp = np.asarray(actor.get_functional_point(0), dtype=float)
            actor_z = float(actor_fp[2])
        except Exception:
            return

        if actor_z > 0.79:
            delay = getattr(self, "audio_delay_time", 0.0)
            if self.scene_side == "left":
                self._play_yes_audio(self.yes_audio_path, "yes_audio_started", "yes_audio_step", delay=delay)
            else:
                self._play_yes_audio(self.question_audio_path, "question_audio_started", "question_audio_step", delay=delay)
