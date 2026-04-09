from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
import numpy as np
import os

class touch_plate_plastic(Base_Task):
    """
    Task: Control the robotic arm to pick up a red box and place it onto target plates sequentially.
    
    Flow:
    1. Load a red box and two target plates (metal_plate and plastic_plate).
    2. Use the LEFT arm to grasp the box.
    3. Place the box onto the 'metal_plate'.
    4. Re-grasp the box and place it onto the 'plastic_plate'.
    5. Return the arm to the origin.
    """

    def setup_demo(self, **kwags):
        """
        Initialize the task environment.
        """
        super()._init_task_env_(**kwags)
        self.metal_audio_path = os.path.join("assets", "audios", "metal.mp3")
        self.plastic_audio_path = os.path.join("assets", "audios", "plastic.mp3")
        self.drop_wait_range = (0.3, 0.7)
        self.metal_audio_started = False
        self.plastic_audio_started = False
        self.metal_audio_step = -1
        self.plastic_audio_step = -1
        

    def _init_plate_assignment(self, assignment):
        valid_positions = {"a", "b"}
        if assignment is None or assignment == "random":
            first = np.random.choice(list(valid_positions))
            second = (valid_positions - {first}).pop()
            return {"metal": first, "plastic": second}
        if not isinstance(assignment, dict):
            raise ValueError("plate_assignment must be a dict or 'random'")
        normalized = {}
        for plate in ("metal", "plastic"):
            pos = assignment.get(plate)
            if not isinstance(pos, str):
                raise ValueError("plate_assignment values must be strings: 'a' or 'b'")
            pos = pos.lower()
            if pos not in valid_positions:
                raise ValueError("plate_assignment positions must be 'a' or 'b'")
            normalized[plate] = pos
        if normalized["metal"] == normalized["plastic"]:
            raise ValueError("metal and plastic must use different positions")
        return normalized

    def load_actors(self):
        """
        Create and load all necessary actors (objects) into the scene.
        """
        # 1. Define pose for the red box (manipulation object)
        # x is fixed at -0.2, z is fixed at 0.842 (table height)
        box_pose = rand_pose(
            xlim=[-0.2, -0.2],
            ylim=[0.0, 0.0],
            zlim=[0.842],
            rotate_rand=False,
        )
        
        # Create the red box
        self.box = create_box(
            scene=self,
            pose=box_pose,
            half_size=(0.02, 0.02, 0.02),
            # half_size=(0.025, 0.025, 0.025),  # ori
            # half_size=(0.025, 0.025, 0.026),
            color=(1, 0, 0),
            name="box",
        )

        self.plate_assignment = self._init_plate_assignment({"metal": "b", "plastic": "a"})
        # self.plate_assignment = self._init_plate_assignment("random")
        self.position_audio_delays = {"a": 0.7, "b": 0.7}
        self.proximity_audio_delays = {"a": 0.0, "b": 0.0}
        if not hasattr(self, "info"):
            self.info = {}
        self.info["plate_assignment"] = dict(self.plate_assignment)

        self.plate_pose_configs = {
            "a": dict(xlim=[-0.13, -0.07], ylim=[-0.03, 0.03]),
            "b": dict(xlim=[-0.03, 0.03], ylim=[-0.13, -0.07]),
        }
        plate_pose_cache = {pos: rand_pose(**cfg) for pos, cfg in self.plate_pose_configs.items()}
        self.plates_by_position = {}
        for plate_type, position in self.plate_assignment.items():
            actor = create_box(
                scene=self,
                pose=plate_pose_cache[position],
                half_size=(0.04, 0.04, 0.005),
                color=(0, 0, 0),
                name=f"{plate_type}_plate",
                is_static=True,
            )
            setattr(self, f"{plate_type}_plate", actor)
            self.plates_by_position[position] = {
                "actor": actor,
                "surface": plate_type,
                "delay": self.position_audio_delays.get(position, 0.0),
                "position": position,
            }
        self.add_prohibit_area(self.box, padding=0.1)
        for info in self.plates_by_position.values():
            self.add_prohibit_area(info["actor"], padding=0.1)

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

    def _play_drop_audio(self, audio_path, started_attr, step_attr, delay=0.0):
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

    def _handle_surface_contact(self, plate_info, delay=None):
        if not plate_info:
            return
        surface_tag = plate_info.get("surface")
        delay = plate_info.get("delay", 0.0) if delay is None else delay
        if surface_tag == "metal":
            self._play_drop_audio(self.metal_audio_path, "metal_audio_started", "metal_audio_step", delay=delay)
        elif surface_tag == "plastic":
            self._play_drop_audio(self.plastic_audio_path, "plastic_audio_started", "plastic_audio_step", delay=delay)

    def _is_box_near_plate(self, box_pos, target_pos, eps_xy=0.04, eps_z=0.02):
        return (
            np.all(np.abs(box_pos[:2] - target_pos[:2]) < eps_xy)
            and abs(box_pos[2] - target_pos[2]) < eps_z
        )

    def _handle_step_audio(self):
        if getattr(self, "audio", None) is None or not getattr(self, "collect_audio", False):
            return
        if not hasattr(self, "box") or not hasattr(self, "plates_by_position"):
            return
        box_pos = self.box.get_functional_point(0, "pose").p
        for info in self.plates_by_position.values():
            surface = info.get("surface")
            if surface not in ("metal", "plastic") or getattr(self, f"{surface}_audio_started", False):
                continue
            target_pos = info["actor"].get_functional_point(1, "pose").p
            if self._is_box_near_plate(box_pos, target_pos):
                position = info.get("position")
                delay = self.proximity_audio_delays.get(position, info.get("delay", 0.0))
                self._handle_surface_contact(info, delay=delay)
                break

    def play_half(self):
        """
        Execute the manipulation sequence.
        """
        # Determine which arm to use (fixed to LEFT arm in this configuration)
        grasp_arm_tag = ArmTag("left")
        target_a = self.plates_by_position["a"]
        target_b = self.plates_by_position["b"]

        # --- STEP 1: Grasp the box ---
        self.move(
            self.grasp_actor(
                self.box,
                arm_tag=grasp_arm_tag,
                pre_grasp_dis=0.09,
                # grasp_dis=0.015,
                
            )
        )

        # --- STEP 2: Lift the box ---
        # Lift by 0.1m to avoid table friction/collision
        self.move(self.move_by_displacement(grasp_arm_tag, z=0.1))

        # --- STEP 3: Place on Metal Plate ---
        # Get target pose from metal_plate's functional point
        # Retain x, y from target, but override z to 0.9 for placement height
        orig_pose_a = target_a["actor"].get_functional_point(1, "pose")
        target_pose_a = sapien.Pose(
            p=[float(orig_pose_a.p[0]), float(orig_pose_a.p[1]), 0.90],
            q=orig_pose_a.q,
        )
        
        self.move(
            self.place_actor(
                self.box,
                target_pose=target_pose_a,
                arm_tag=grasp_arm_tag,
                functional_point_id=0,
                pre_dis=0,
                dis=0,
                is_open=False,
                constrain="free",
            )
        )
        # self._handle_surface_contact(target_a)
        self.move(self.open_gripper(grasp_arm_tag))
        # left_pose = np.asarray(self.get_arm_pose("left"), dtype=float)
        self.delay(3)

    def play_once(self):
        """
        Execute the manipulation sequence.
        """
        # Determine which arm to use (fixed to LEFT arm in this configuration)
        grasp_arm_tag = ArmTag("left")
        target_a = self.plates_by_position["a"]
        target_b = self.plates_by_position["b"]
        self.set_stage_label("grasp_box")
        # --- STEP 1: Grasp the box ---
        self.move(
            self.grasp_actor(
                self.box,
                arm_tag=grasp_arm_tag,
                pre_grasp_dis=0.09,
                # grasp_dis=0.015,
                
            )
        )

        # --- STEP 2: Lift the box ---
        # Lift by 0.1m to avoid table friction/collision
        self.set_stage_label("lift_box")
        self.move(self.move_by_displacement(grasp_arm_tag, z=0.1))

        # --- STEP 3: Place on Metal Plate ---
        # Get target pose from metal_plate's functional point
        # Retain x, y from target, but override z to 0.9 for placement height
        orig_pose_a = target_a["actor"].get_functional_point(1, "pose")
        target_pose_a = sapien.Pose(
            p=[float(orig_pose_a.p[0]), float(orig_pose_a.p[1]), 0.90],
            q=orig_pose_a.q,
        )
        self.set_stage_label("put_on_a")
        self.move(
            self.place_actor(
                self.box,
                target_pose=target_pose_a,
                arm_tag=grasp_arm_tag,
                functional_point_id=0,
                pre_dis=0,
                dis=0,
                is_open=False,
                constrain="free",
            )
        )
        self._handle_surface_contact(target_a)
        self.move(self.open_gripper(grasp_arm_tag))
        # left_pose = np.asarray(self.get_arm_pose("left"), dtype=float)
        # self._wait_and_update(1.0, save_freq=self.save_freq)

        placed_on_plastic = target_a["surface"] == "plastic"

        if not placed_on_plastic:
            self.set_stage_label("grasp_box_again")
            # --- STEP 4: Re-grasp the box ---
            self.move(
                self.grasp_actor(
                    self.box,
                    arm_tag=grasp_arm_tag,
                    pre_grasp_dis=0.09,
                    grasp_dis=0.015,
                )
            )
            # --- STEP 5: Place on Plastic Plate ---
            orig_pose_b = target_b["actor"].get_functional_point(1, "pose")
            target_pose_b = sapien.Pose(
                p=[float(orig_pose_b.p[0]), float(orig_pose_b.p[1]), 0.90],
                q=orig_pose_b.q,
            )
            self.set_stage_label("put_on_b")
            self.move(
                self.place_actor(
                    self.box,
                    target_pose=target_pose_b,
                    arm_tag=grasp_arm_tag,
                    functional_point_id=0,
                    pre_dis=0,
                    dis=0,
                    is_open=False,
                    constrain="free",
                )
            )
            self._handle_surface_contact(target_b)
            self.move(self.open_gripper(grasp_arm_tag))
            # left_pose = np.asarray(self.get_arm_pose("left"), dtype=float)
        # --- STEP 6: Return to origin ---
        self.set_stage_label("return_to_origin")
        self.move(self.back_to_origin(grasp_arm_tag))

        self.info["audio"] = {
            "metal_audio_started": self.metal_audio_started,
            "metal_audio_step": self.metal_audio_step,
            "plastic_audio_started": self.plastic_audio_started,
            "plastic_audio_step": self.plastic_audio_step,
            "position_audio_delays": dict(getattr(self, "position_audio_delays", {})),
        }
        self.info["plate_assignment"] = dict(self.plate_assignment)
        return self.info

    def check_success(self):
        plastic_pos = self.plate_assignment.get("plastic") if hasattr(self, "plate_assignment") else None
        if plastic_pos is None or not hasattr(self, "plates_by_position"):
            return False
        plastic_actor = self.plates_by_position.get(plastic_pos, {}).get("actor")
        if plastic_actor is None:
            return False

        box_pos = self.box.get_functional_point(0, "pose").p
        target_pos = plastic_actor.get_functional_point(1, "pose").p
        eps_xy, eps_z = 0.04, 0.01
        box_on_plastic = (
            np.all(np.abs(box_pos[:2] - target_pos[:2]) < eps_xy)
            and abs(box_pos[2] - target_pos[2]) < eps_z
        )

        # If plastic is placed at position b, placing the box on the plastic plate is sufficient.
        if plastic_pos == "b":
            return bool(box_on_plastic)

        # Otherwise, keep the stricter success check.
        if not box_on_plastic:
            return False

        robot = getattr(self, "robot", None)
        if robot is None or not hasattr(robot, "left_original_pose"):
            return False

        put_a_pose = np.asarray(
            [-0.09957024455070496, 0.0015785980504006147, 1.0555778741836548],
            dtype=float,
        )
        left_pose = np.asarray(self.get_arm_pose("left"), dtype=float)
        dist_from_a = np.linalg.norm(left_pose[:3] - put_a_pose[:3])
        return dist_from_a > 0.2

        # origin_pose = np.asarray(robot.left_original_pose, dtype=float)
        # left_pose = np.asarray(self.get_arm_pose("left"), dtype=float)
        # pos_close = np.linalg.norm(left_pose[:3] - origin_pose[:3]) < 0.01
        # rot_error_deg = cal_quat_dis(left_pose[3:], origin_pose[3:]) * 180
        # rot_close = rot_error_deg < 5.0
        # return pos_close and rot_close
