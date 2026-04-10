import dataclasses

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


def make_SanyaDroid_example() -> dict:
    return {
        "state": np.ones((8,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class SanyaDroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)

        stage = data.get("stage")
        if stage is not None:

            if stage == "random_wait" and "actions" in data:
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


        base_image = _parse_image(data["images"]["cam_high"])
        wrist_image = _parse_image(data["images"]["cam_left_wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")


        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        if "stage" in data:
            if isinstance(data["stage"], bytes):
                data["stage"] = data["stage"].decode("utf-8")
            inputs["stage"] = data["stage"]

        if "audio" in data:
            inputs["audio"] = data["audio"]

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
class SanyaDroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}
