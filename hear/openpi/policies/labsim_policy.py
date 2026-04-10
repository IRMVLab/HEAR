import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LabSimInputs(transforms.DataTransformFn):

    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # Parse available cameras
        camera_1_rgb = _parse_image(data["camera_1_rgb"]) if "camera_1_rgb" in data else None
        camera_2_rgb = _parse_image(data["camera_2_rgb"]) if "camera_2_rgb" in data else None
        camera_3_rgb = _parse_image(data["camera_3_rgb"]) if "camera_3_rgb" in data else None

        # Get a reference image for creating zero placeholders
        reference_image = camera_1_rgb if camera_1_rgb is not None else (
            camera_2_rgb if camera_2_rgb is not None else camera_3_rgb
        )

        # Use actual images or zero placeholders
        base_image = camera_1_rgb if camera_1_rgb is not None else np.zeros_like(reference_image)
        left_wrist_image = camera_2_rgb if camera_2_rgb is not None else np.zeros_like(reference_image)
        right_wrist_image = camera_3_rgb if camera_3_rgb is not None else np.zeros_like(reference_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_ if camera_1_rgb is not None else (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                ),
                "left_wrist_0_rgb": np.True_ if camera_2_rgb is not None else (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                ),
                "right_wrist_0_rgb": np.True_ if camera_3_rgb is not None else (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                ),
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LabSimOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}