import dataclasses
from typing import ClassVar

import einops
import numpy as np
import torch

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

class VLAStateAugmenter:
    def __init__(self, augment_prob=0.9):

        self.augment_prob = augment_prob


        self.noise_std_vector = np.array([
            0.01,
            0.05,
            0.04,
            0.01,
            0.01,
            0.01,
            0.025,

            0.01,
            0.05,
            0.04,
            0.01,
            0.01,
            0.01,
            0.025
        ], dtype=np.float32)


        self.min_limits = np.ones(14) * -6.28
        self.max_limits = np.ones(14) * 6.28


        self.min_limits[6] = 0.0
        self.max_limits[6] = 1.0
        self.min_limits[13] = 0.0
        self.max_limits[13] = 1.0

    def add_noise(self, state):

        if np.random.rand() > self.augment_prob:
            return state


        noise = np.random.normal(loc=0.0, scale=self.noise_std_vector)


        noisy_state = state + noise


        noisy_state = np.clip(noisy_state, self.min_limits, self.max_limits)

        return noisy_state.astype(np.float32)

@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    state_augmenter: ClassVar[VLAStateAugmenter] = VLAStateAugmenter(augment_prob=0.9)

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)


        state = data["state"]
        stage = data.get("stage")
        if stage is not None:
        # if 0:

            if (stage == "wait" or stage == "padding") and "actions" in data:
                actions_arr = data["actions"]

                if torch.is_tensor(actions_arr):
                    n_rows = 1 if actions_arr.ndim == 1 else actions_arr.size(0)
                    state_tensor = torch.as_tensor(state, dtype=actions_arr.dtype, device=actions_arr.device)
                    data["actions"] = state_tensor.unsqueeze(0).repeat(n_rows, 1)
                else:
                    actions_arr = np.asarray(actions_arr)
                    n_rows = 1 if actions_arr.ndim == 1 else actions_arr.shape[0]
                    state_np = np.asarray(state, dtype=actions_arr.dtype)
                    data["actions"] = np.repeat(state_np[None, :], n_rows, axis=0)


        # if actions is not None:

        #     state_sum = state.sum() if torch.is_tensor(state) else np.sum(state)
        #     if np.isclose(state_sum, 2.0, atol=1e-5):
        #         if torch.is_tensor(actions):
        #             if actions.ndim >= 2 and actions.size(0) >= 3:
        #                 state_tensor = torch.as_tensor(state, dtype=actions.dtype, device=actions.device)

        #                 target_chunk = state_tensor.unsqueeze(0).expand(3, -1)

        #                 if torch.allclose(actions[:3], target_chunk, atol=1e-5):

        #                     data["actions"] = state_tensor.unsqueeze(0).repeat(actions.size(0), 1)
        #                     actions = data["actions"]
        #         else:
        #             actions_np = np.asarray(actions)
        #             if actions_np.ndim >= 2 and actions_np.shape[0] >= 3:
        #                 state_np = np.asarray(state)
        #                 target_chunk = np.repeat(state_np[None, :], 3, axis=0)

        #                 if np.allclose(actions_np[:3], target_chunk, atol=1e-5):
        #                     tiled = np.repeat(state_np[None, :], actions_np.shape[0], axis=0)
        #                     data["actions"] = tiled
        #                     actions = data["actions"]
        # # !!!!!!!!!!!!!!!!!!!!!!!!!!!

        # row_sums = data["actions"].sum(dim=1)
        # if not torch.allclose(row_sums, torch.full_like(row_sums, 2.0), atol=1e-5):

        # total = data["state"].sum()
        # if not np.isclose(total, 2.0, atol=1e-5):

        # # !!!!!!!!!!!!!!!!!!!!!!!!!!!

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # state = data["state"]

        # noisy_state = self.state_augmenter.add_noise(state)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "stage" in data:
            inputs["stage"] = data["stage"]

        if "audio" in data:
            inputs["audio"] = data["audio"]

        if "hist_audio" in data:
            inputs["hist_audio"] = data["hist_audio"]

        if "next_audio" in data:
            inputs["next_audio"] = data["next_audio"]

        if "timestamp" in data:
            inputs["timestamp"] = data["timestamp"]

        if "frame_index" in data:
            inputs["frame_index"] = data["frame_index"]

        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]

        if "index" in data:
            inputs["index"] = data["index"]

        if "task_index" in data:
            inputs["task_index"] = data["task_index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
