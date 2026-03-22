"""Random-policy baseline using OffPolicyRunner.

Runs the full rsl_rl runner infrastructure (logging, progress.csv,
TensorBoard, model saving) with RandomPolicy — no gradient updates.

The log directory format mirrors training runs so progress.csv can be
overlaid directly on a learning-curve plot.

Usage:
    python legged_gym/scripts/train_random.py --task=go2_flat --headless
    python legged_gym/scripts/train_random.py --task=go2_flat_sac --headless
"""
import os
from datetime import datetime

import isaacgym  # must be imported before torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args, class_to_dict

from rsl_rl.runners import OffPolicyRunner


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    _, train_cfg = task_registry.get_cfgs(name=args.task)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    # Override the algorithm to RandomPolicy; keep everything else
    # (num_steps_per_env, save_interval, max_iterations, logging) intact.
    train_cfg_dict = class_to_dict(train_cfg)
    train_cfg_dict["runner"]["algorithm_class_name"] = "RandomPolicy"
    train_cfg_dict["runner"]["run_name"] = "random"

    experiment_name = train_cfg_dict["runner"].get("experiment_name", args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name)
    log_dir = os.path.join(
        log_root,
        datetime.now().strftime("%b%d_%H-%M-%S") + "_random",
    )

    env.reset()
    runner = OffPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)

    print(f"\nRandom policy baseline  |  task={args.task}  |  num_envs={env.num_envs}")
    print(f"Log dir: {log_dir}\n")

    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)


if __name__ == "__main__":
    args = get_args()
    train(args)
