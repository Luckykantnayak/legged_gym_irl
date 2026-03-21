import argparse
import glob
import os
import sys
import tempfile

import isaacgym
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, Logger
from legged_gym.utils.helpers import get_load_path, class_to_dict, update_cfg_from_args
from legged_gym import LEGGED_GYM_ROOT_DIR

from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner

# Parse recording args and strip them from sys.argv so get_args() / gymutil
# (which calls parse_args() strictly) doesn't see unrecognised flags.
_rec_parser = argparse.ArgumentParser(add_help=False)
_rec_parser.add_argument("--record", action="store_true")
_rec_parser.add_argument("--video_dir",  type=str, default="videos/")
_rec_parser.add_argument("--video_name", type=str, default="episode.mp4")
_rec_parser.add_argument("--fps",        type=int, default=30)
# viewer camera position and look-at target
_rec_parser.add_argument("--cam_pos",    type=float, nargs=3, default=None,
                         metavar=("X", "Y", "Z"),
                         help="Camera position in world space, e.g. --cam_pos -2.5 0 1.35")
_rec_parser.add_argument("--cam_lookat", type=float, nargs=3, default=None,
                         metavar=("X", "Y", "Z"),
                         help="Camera look-at target, e.g. --cam_lookat 0 0 0.35")
_rec_parser.add_argument("--num_envs", type=int, default=50,
                         help="Number of environments to render (default: 50)")
_rec_args, _remaining = _rec_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining  # hide recording flags from get_args()


def play(args):
    rec_args = _rec_args

    # Recording uses the viewer (write_viewer_image_to_file).
    # A display is required — run on DCV / with a monitor attached.
    if rec_args.record:
        args.headless = False

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, rec_args.num_envs)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    print(f"Loading model from: {resume_path}")

    train_cfg_dict = class_to_dict(train_cfg)
    alg_name = train_cfg_dict["runner"].get("algorithm_class_name", "SAC")
    if alg_name == "PPO":
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device=args.rl_device)
    else:
        runner = OffPolicyRunner(env, train_cfg_dict, log_dir=None, device=args.rl_device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1

    # --- recording setup ---
    frame_dir = None
    frame_idx = 0
    if rec_args.record:
        if env.viewer is None:
            print("WARNING: viewer is None — cannot record. Re-run without --headless.")
        else:
            frame_dir = tempfile.mkdtemp(prefix="legged_frames_")
            print(f"Recording frames to tmp dir: {frame_dir}")
            print(f"Output video: {os.path.join(rec_args.video_dir, rec_args.video_name)}")

    # --- set viewer camera if position args were given ---
    if env.viewer is not None and rec_args.cam_pos is not None:
        from isaacgym import gymapi as _gymapi
        cp = rec_args.cam_pos
        lt = rec_args.cam_lookat if rec_args.cam_lookat is not None else [0.0, 0.0, 0.35]
        env.gym.viewer_camera_look_at(
            env.viewer, None,
            _gymapi.Vec3(*cp),
            _gymapi.Vec3(*lt),
        )

    obs = env.get_observations()
    for i in range(1 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # --- capture frame via viewer ---
        if frame_dir is not None and env.viewer is not None:
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
            env.gym.write_viewer_image_to_file(env.viewer, frame_path)
            frame_idx += 1

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

    # --- stitch frames into video ---
    if frame_dir is not None and frame_idx > 0:
        try:
            import imageio
            png_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
            frames = [imageio.imread(p) for p in png_files]
            os.makedirs(rec_args.video_dir, exist_ok=True)
            out_path = os.path.join(rec_args.video_dir, rec_args.video_name)
            imageio.mimsave(out_path, frames, fps=rec_args.fps, macro_block_size=None)
            print(f"Saved video: {out_path}  ({len(frames)} frames)")
        except ImportError:
            print("imageio not installed. Run: pip install imageio imageio-ffmpeg")
        finally:
            # clean up temp PNGs
            import shutil
            shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == '__main__':
    args = get_args()
    play(args)
