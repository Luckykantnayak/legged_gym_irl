import sys
import os

# Use laser_rsl_rl (which provides LeggedGymRunner + SAC) instead of the default rsl_rl
LASER_RSL_RL_PATH = os.path.join(os.path.dirname(__file__), '../../../laser_rsl_rl')
sys.path.insert(0, os.path.abspath(LASER_RSL_RL_PATH))

import isaacgym  # must be imported before torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args, class_to_dict
from legged_gym import LEGGED_GYM_ROOT_DIR

from rsl_rl.runners import LeggedGymRunner
from datetime import datetime


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    _, train_cfg = task_registry.get_cfgs(name=args.task)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

    train_cfg_dict = class_to_dict(train_cfg)
    runner = LeggedGymRunner(env, train_cfg_dict, log_dir, device=args.rl_device)

    if train_cfg.runner.resume:
        from legged_gym.utils.helpers import get_load_path
        resume_path = get_load_path(
            log_root,
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint,
        )
        print(f"Loading model from: {resume_path}")
        runner.load(resume_path)

    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)


if __name__ == '__main__':
    args = get_args()
    train(args)
