import os

import numpy as np

from ._base_task import Base_Task
from .utils import *


class shake_bottle_audio(Base_Task):
    POSE1_TARGET = np.array([-0.20858605, -0.17626616, 1.01734901], dtype=float)
    POSE2_TARGET = np.array([0.08890237, -0.17493004, 1.01790822], dtype=float)
    POSE_TOLERANCE = 0.04

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)
        scenario = "full"
        self.scenario = self._init_scenario(scenario)
        self.shake_audio_path = os.path.join("assets", "audios", "bottle_shake.mp3")
        self._shake_audio_started = False
        self._shake_audio_stopped = False

    def _init_scenario(self, scenario):
        valid = {"full", "empty"}
        if scenario is None or str(scenario).lower() == "random":
            return np.random.choice(list(valid))
        scenario = str(scenario).lower()
        if scenario not in valid:
            raise ValueError("scenario must be 'full' or 'empty'")
        return scenario

    def load_actors(self):
        # self.id_list = [i for i in range(20)]
        rand_pos = rand_pose(
            xlim=[-0.2, -0.2],
            ylim=[-0.3, -0.3],
            zlim=[0.785],
            qpos=[1.0, 0.0, 0.0, 0.0],
            rotate_rand=False,
            # rotate_lim=[0, 0, np.pi / 4],
        )
        # while abs(rand_pos.p[0]) < 0.1:
        #     rand_pos = rand_pose(
        #         xlim=[-0.15, 0.15],
        #         ylim=[-0.15, -0.05],
        #         zlim=[0.785],
        #         qpos=[0, 0, 1, 0],
        #         rotate_rand=True,
        #         rotate_lim=[0, 0, np.pi / 4],
        #     )
        # self.bottle_id = np.random.choice(self.id_list)
        self.bottle_id = 13
        self.bottle = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="001_bottle",
            convex=True,
            model_id=self.bottle_id,
        )
        self.bottle.set_mass(0.01)
        self.add_prohibit_area(self.bottle, padding=0.05)


        rand_pos = rand_pose(
            xlim=[-0.2, -0.2],
            ylim=[0.1, 0.1],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            # rotate_lim=[0, 3.14, 0],
        )
        self.basket_id = 3
        self.breadbasket_1 = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="076_breadbasket",
            convex=True,
            model_id=self.basket_id,
        )
        self.add_prohibit_area(self.breadbasket_1, padding=0.05)

        rand_pos = rand_pose(
            xlim=[0.0, 0.0],
            ylim=[0.1, 0.1],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            # rotate_lim=[0, 3.14, 0],
        )
        self.breadbasket_2 = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="076_breadbasket",
            convex=True,
            model_id=self.basket_id,
        )
        self.add_prohibit_area(self.breadbasket_2, padding=0.05)

    def play_once(self):
        arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(self.bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
        upright_quat = [0.707, 0, 0, 0.707]
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, quat=upright_quat))
        y_rotation = t3d.euler.euler2quat(0, (np.pi / 4) * 3, 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, upright_quat)
        tilted_quat = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        if self.scenario == "full" and getattr(self, "collect_audio", False) and getattr(self, "audio", None):
            self.audio.start_playing(
                self.shake_audio_path,
                loop_audio=True,
                randomize_start=True,
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.1, y=0.0, quat=tilted_quat))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.2, y=0.0, quat=upright_quat))

        if self.scenario == "full" and getattr(self, "collect_audio", False) and getattr(self, "audio", None):
            self.audio.stop_playing()

        target_basket = self.breadbasket_1 if self.scenario == "full" else self.breadbasket_2
        target_pose = target_basket.get_functional_point(0)
        self.move(
            self.place_actor(
                self.bottle,
                arm_tag=arm_tag,
                target_pose=target_pose,
                constrain="free",
                pre_dis=0.12,
                is_open=False,
            )
        )
        self.move(self.open_gripper(arm_tag=arm_tag))

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle_id}",
            "{a}": str(arm_tag),
            # "{scenario}": self.scenario,
        }
        return self.info

    def _arm_pose_near(self, pose, target):
        if pose is None:
            return False
        arr = np.asarray(pose, dtype=float)
        if arr.size < 3:
            return False
        return np.linalg.norm(arr[:3] - target) <= self.POSE_TOLERANCE

    def _handle_step_audio(self):
        if self.scenario != "full" or not getattr(self, "collect_audio", False):
            return
        audio = getattr(self, "audio", None)
        if audio is None or self._shake_audio_stopped:
            return

        near_pose1 = False
        near_pose2 = False
        for tag in ("left", "right"):
            try:
                pose = self.get_arm_pose(tag)
            except Exception:
                continue
            near_pose1 = near_pose1 or self._arm_pose_near(pose, self.POSE1_TARGET)
            near_pose2 = near_pose2 or self._arm_pose_near(pose, self.POSE2_TARGET)

        if not self._shake_audio_started and near_pose1:
            started = audio.start_playing(self.shake_audio_path, loop_audio=True, randomize_start=True)
            if started:
                self._shake_audio_started = True
        elif self._shake_audio_started and not self._shake_audio_stopped and near_pose2:
            audio.stop_playing()
            self._shake_audio_stopped = True

    def check_success(self):
        bottle_pos = self.bottle.get_pose().p
        basket1_pos = self.breadbasket_1.get_pose().p
        basket2_pos = self.breadbasket_2.get_pose().p
        dist_to_basket1 = np.sqrt((bottle_pos[0] - basket1_pos[0]) ** 2 + (bottle_pos[1] - basket1_pos[1]) ** 2)
        dist_to_basket2 = np.sqrt((bottle_pos[0] - basket2_pos[0]) ** 2 + (bottle_pos[1] - basket2_pos[1]) ** 2)
        threshold = 0.15
        if self.scenario == "full":
            return dist_to_basket1 < threshold
        return dist_to_basket2 < threshold
