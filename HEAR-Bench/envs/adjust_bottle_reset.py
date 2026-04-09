from ._base_task import Base_Task
from .utils import *
import sapien
import math
import os
import numpy as np


class adjust_bottle_reset(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.reset_audio_path = os.path.join("assets", "audios", "reset.mp3")

    def load_actors(self):
        self.qpose_tag = np.random.randint(0, 2)
        qposes = [[0.707, 0.0, 0.0, -0.707], [0.707, 0.0, 0.0, 0.707]]
        xlims = [[-0.12, -0.08], [0.08, 0.12]]

        self.model_id = np.random.choice([13, 16])

        self.bottle = rand_create_actor(
            self,
            xlim=xlims[self.qpose_tag],
            ylim=[-0.13, -0.08],
            zlim=[0.752],
            rotate_rand=True,
            qpos=qposes[self.qpose_tag],
            modelname="001_bottle",
            convex=True,
            rotate_lim=(0, 0, 0.4),
            model_id=self.model_id,
        )
        self.delay(4)
        self.add_prohibit_area(self.bottle, padding=0.15)
        self.left_target_pose = [-0.25, -0.12, 0.95, 0, 1, 0, 0]
        self.right_target_pose = [0.25, -0.12, 0.95, 0, 1, 0, 0]
        self.reset_audio_started = False
        self.audio_finished_interrupt = False

    def play_once(self):
        self.set_stage_label("wait")
        self.audio_finished_interrupt = False

        delay = np.random.uniform(0.5, 7.0)
        if self.collect_audio and hasattr(self, "audio"):
            self.audio.start_playing(
                self.reset_audio_path,
                loop_audio=False,
                randomize_start=False,
                delay=delay,
            )

        arm_tag = ArmTag("right" if self.qpose_tag == 1 else "left")
        target_pose = (self.right_target_pose if self.qpose_tag == 1 else self.left_target_pose)

        if not self.audio_finished_interrupt:
            self.set_stage_label("grasp")
            self.move(
                self.grasp_actor(self.bottle, arm_tag=arm_tag, pre_grasp_dis=0.1),
                interrupt_on_audio_finish=True
            )
        
        if not self.audio_finished_interrupt:
            self.set_stage_label("lift")
            self.move(
                self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"),
                interrupt_on_audio_finish=True
            )
        
        if not self.audio_finished_interrupt:
            self.set_stage_label("show")
            self.move(
                self.place_actor(
                    self.bottle,
                    target_pose=target_pose,
                    arm_tag=arm_tag,
                    functional_point_id=0,
                    pre_dis=0.0,
                    is_open=False,
                ),
                interrupt_on_audio_finish=True
            )

        if self.audio_finished_interrupt:
            self.set_stage_label("interrupt")
            self.together_open_gripper()
            self.move(self.back_to_origin(arm_tag), interrupt_on_audio_finish=False)

            self.info["info"] = {
                "{A}": f"001_bottle/base{self.model_id}",
                "{a}": str(arm_tag),
            }
            return self.info

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        """
        The selected arm must return close to its original end-effector pose,
        and at least one audio clip must have been played when audio collection is enabled.
        """
        pos_thresh = 0.05
        quat_thresh = 0.15

        arm_is_right = (self.qpose_tag == 1)
        if arm_is_right:
            current_pose = np.array(self.robot.get_right_ee_pose())
            origin_pose = np.array(self.robot.right_original_pose)
        else:
            current_pose = np.array(self.robot.get_left_ee_pose())
            origin_pose = np.array(self.robot.left_original_pose)

        pos_err = np.linalg.norm(current_pose[:3] - origin_pose[:3])

        # try:
        #     quat_err = cal_quat_dis(current_pose[3:], origin_pose[3:])
        # except Exception:
        #     quat_err = np.linalg.norm(current_pose[3:] - origin_pose[3:])

        audio_ok = True
        if getattr(self, "collect_audio", False) and hasattr(self, "audio") and self.audio is not None:
            audio_ok = getattr(self.audio, "play_count", 0) > 0

        return (pos_err <= pos_thresh) and audio_ok


    def _handle_step_audio(self):
        if getattr(self, "audio", None) is None or not getattr(self, "collect_audio", False):
            return

        if getattr(self, "reset_audio_started", False):
            return

        take_cnt = getattr(self, "take_action_cnt", None)
        if take_cnt is None:
            return

        if int(take_cnt) == 180:
            try:
                self.reset_audio_started = True
                self.audio.start_playing(
                    self.reset_audio_path,
                    loop_audio=False,
                    randomize_start=False,
                    delay=0.0
                )
            except Exception:
                self.reset_audio_started = False
