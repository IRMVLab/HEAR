from ._base_task import Base_Task
from .utils import *
import sapien
import math
import os


class lift_pot_audio(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)
        self.boiling_audio_path = os.path.join("assets", "audios", "boiling_water.mp3")

    def load_actors(self):
        self.model_name = "060_kitchenpot"
        self.model_id = np.random.randint(0, 2)
        self.pot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[-0.05, 0.05],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
            qpos=[0.704141, 0, 0, 0.71006],
        )
        x, y = self.pot.get_pose().p[0], self.pot.get_pose().p[1]
        self.prohibited_area.append([x - 0.3, y - 0.1, x + 0.3, y + 0.1])

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

    def _handle_step_audio(self):
        if getattr(self, "audio", None) is None or not getattr(self, "collect_audio", False):
            return

        if getattr(self, "boiling_audio_started", False):
            return

        take_cnt = getattr(self, "take_action_cnt", None)
        if take_cnt is None:
            return

        step_target = int(getattr(self, "boiling_audio_step", 30))
        if int(take_cnt) == step_target:
            try:
                self.boiling_audio_started = True
                self.audio.start_playing(
                    self.boiling_audio_path,
                    loop_audio=False,
                    randomize_start=False,
                    delay=0.0,
                )
            except Exception:
                self.boiling_audio_started = False

    def play_once(self):
        self.set_stage_label("wait")
        post_audio_wait = 8.1
        self._wait_and_update(3.0, save_freq=self.save_freq)
        if self.collect_audio and hasattr(self, "audio"):
            self.audio.start_playing(
                self.boiling_audio_path,
                loop_audio=False,
                randomize_start=False,
            )
        self._wait_and_update(post_audio_wait, save_freq=self.save_freq)
        self.set_stage_label("move")
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")
        # Close both left and right grippers to half position
        self.move(
            self.close_gripper(left_arm_tag, pos=0.5),
            self.close_gripper(right_arm_tag, pos=0.5),
        )
        # Grasp the pot with both arms at specified contact points
        self.move(
            self.grasp_actor(self.pot, left_arm_tag, pre_grasp_dis=0.035, contact_point_id=0),
            self.grasp_actor(self.pot, right_arm_tag, pre_grasp_dis=0.035, contact_point_id=1),
        )
        self.set_stage_label("lift")
        # Lift the pot by moving both arms upward to target height (0.88)
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.88 - self.pot.get_pose().p[2]),
            self.move_by_displacement(right_arm_tag, z=0.88 - self.pot.get_pose().p[2]),
        )

        self.info["info"] = {"{A}": f"{self.model_name}/base{self.model_id}"}
        return self.info

    def check_success(self):
        pot_pose = self.pot.get_pose()
        left_end = np.array(self.robot.get_left_tcp_pose()[:3])
        right_end = np.array(self.robot.get_right_tcp_pose()[:3])
        left_grasp = np.array(self.pot.get_contact_point(0)[:3])
        right_grasp = np.array(self.pot.get_contact_point(1)[:3])
        pot_dir = get_face_prod(pot_pose.q, [0, 0, 1], [0, 0, 1])
        return (pot_pose.p[2] > 0.82 and np.sqrt(np.sum((left_end - left_grasp)**2)) < 0.03
                and np.sqrt(np.sum((right_end - right_grasp)**2)) < 0.03 and pot_dir > 0.8)
