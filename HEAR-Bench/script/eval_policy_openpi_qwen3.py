"""Legacy compatibility wrapper for HEAR-Bench pi0 evaluation."""

from eval_policy import main, parse_args_and_config


DEFAULTS = {
    "config": "policy/pi0/deploy_policy.yml",
    "task_name": "click_alarmclock_audio_random",
    "task_config": "demo_clean_audio",
    "train_config_name": "robotwin",
    "ckpt_setting": "pi0",
    "policy_name": "pi0",
    "instruction_type": "seen",
    "checkpoint_id": "30000",
    "pi0_step": 50,
    "visualize": False,
    "execute_size": 32,
}


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    main(parse_args_and_config(DEFAULTS))
