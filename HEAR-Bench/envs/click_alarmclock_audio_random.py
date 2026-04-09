import os

import numpy as np

from ._base_task import Base_Task
from .utils import *


class click_alarmclock_audio_random(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.alarm_audio_path = os.path.join(
            "assets",
            "audios",
            "alarm_clock.mp3",
        )


    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        while abs(rand_pos.p[0]) < 0.05:
            rand_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )

        self.alarmclock_id = np.random.choice([1, 3], 1)[0]
        self.alarm = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="046_alarm-clock",
            convex=True,
            model_id=self.alarmclock_id,
            is_static=True,
        )
        self.add_prohibit_area(self.alarm, padding=0.05)
        self.check_arm_function = self.is_left_gripper_close if self.alarm.get_pose().p[0] < 0 else self.is_right_gripper_close

    def _wait_and_update(self, duration, save_freq=None):
        """Wait for a duration while keeping audio and capture state updated."""
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
        arm_tag = ArmTag("right" if self.alarm.get_pose().p[0] > 0 else "left")

        self.move((
            ArmTag(arm_tag),
            [   
                Action(
                    arm_tag,
                    "move",
                    self.get_grasp_pose(self.alarm, pre_dis=0.1, contact_point_id=0, arm_tag=arm_tag)[:3] +
                    [0.5, -0.5, 0.5, 0.5],
                ),
                Action(arm_tag, "close", target_gripper_pos=0.0),
            ],
        ))

    def play_once(self):
        arm_tag = ArmTag("right" if self.alarm.get_pose().p[0] > 0 else "left")
        self.set_stage_label("wait")
        random_wait_time = float(np.random.uniform(1.0, 5.0))
        self._wait_and_update(random_wait_time, save_freq=self.save_freq)
        if self.collect_audio and hasattr(self, "audio"):
            self.audio.start_playing(
                self.alarm_audio_path,
                loop_audio=True,
                randomize_start=True,
            )
        self.set_stage_label("alarm")
        random_reaction_time = np.random.uniform(0.6, 2.0)
        self._wait_and_update(random_reaction_time, save_freq=self.save_freq)
        self.set_stage_label("press")
        self.move((
            ArmTag(arm_tag),
            [   
                Action(
                    arm_tag,
                    "move",
                    self.get_grasp_pose(self.alarm, pre_dis=0.1, contact_point_id=0, arm_tag=arm_tag)[:3] +
                    [0.5, -0.5, 0.5, 0.5],
                ),
                Action(arm_tag, "close", target_gripper_pos=0.0),
            ],
        ))
        self.set_stage_label("lift")
        self.move(self.move_by_displacement(arm_tag, z=-0.065))
        
        self.info["info"] = {
            "{A}": f"046_alarm-clock/base{self.alarmclock_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        if self.stage_success_tag:
            return True

        if not self.check_arm_function():
            return False

        alarm_pose = self.alarm.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("046_alarm-clock")
        eps = [0.01, 0.01]
        for position in positions:
            if (np.all(np.abs(position[:2] - alarm_pose[:2]) < eps) and
                abs(position[2] - alarm_pose[2]) < 0.01):
                self.stage_success_tag = True
                return True
        return False

    def _handle_step_audio(self):
        if getattr(self, "audio", None) is None or not getattr(self, "collect_audio", False):
            return

        if getattr(self, "alarm_audio_started", False):
            return

        take_cnt = getattr(self, "take_action_cnt", None)
        if take_cnt is None:
            return

        if int(take_cnt) == 200:
            try:
                self.alarm_audio_started = True
                self.audio.start_playing(
                    self.alarm_audio_path,
                    loop_audio=False,
                    randomize_start=False,
                    delay=0.0
                )
            except Exception:
                self.alarm_audio_started = False
