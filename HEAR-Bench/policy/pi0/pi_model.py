from pathlib import Path

import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PI0:
    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        checkpoint_dir = Path("policy/pi0/checkpoints") / self.train_config_name / self.model_name / str(self.checkpoint_id)
        self.policy = _policy_config.create_trained_policy(
            config,
            str(checkpoint_dir),
            robotwin_repo_id=model_name,
        )
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.instruction = None

    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language(self, instruction):
        self.instruction = instruction

    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = img_arr
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
