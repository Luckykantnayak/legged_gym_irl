import argparse
import glob
import os
import sys
import tempfile

import isaacgym
from isaacgym import gymapi
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, Logger

# Parse extra args and strip them from sys.argv before get_args() / gymutil.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--record", action="store_true")
_parser.add_argument("--video_dir",  type=str, default="videos/")
_parser.add_argument("--video_name", type=str, default="random.mp4")
_parser.add_argument("--fps",        type=int, default=30)
_parser.add_argument("--cam_pos",    type=float, nargs=3, default=None,
                     metavar=("X", "Y", "Z"),
                     help="Camera position, e.g. --cam_pos -2.5 0 1.35")
_parser.add_argument("--cam_lookat", type=float, nargs=3, default=None,
                     metavar=("X", "Y", "Z"),
                     help="Camera look-at target, e.g. --cam_lookat 0 0 0.35")
_parser.add_argument("--num_envs",   type=int, default=50,
                     help="Number of environments to render (default: 50)")
_extra_args, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining


def play(args):
    extra = _extra_args

    if extra.record:
        args.headless = False

    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, extra.num_envs)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1

    # --- recording setup ---
    frame_dir = None
    frame_idx = 0
    if extra.record:
        if env.viewer is None:
            print("WARNING: viewer is None — cannot record. Re-run without --headless.")
        else:
            frame_dir = tempfile.mkdtemp(prefix="legged_frames_")
            print(f"Recording frames to tmp dir: {frame_dir}")
            print(f"Output video: {os.path.join(extra.video_dir, extra.video_name)}")

    # --- set viewer camera ---
    if env.viewer is not None and extra.cam_pos is not None:
        cp = extra.cam_pos
        lt = extra.cam_lookat if extra.cam_lookat is not None else [0.0, 0.0, 0.35]
        env.gym.viewer_camera_look_at(
            env.viewer, None,
            gymapi.Vec3(*cp),
            gymapi.Vec3(*lt),
        )

    obs = env.get_observations()
    for i in range(int(env.max_episode_length)):
        actions = torch.rand(env.num_envs, env.num_actions, device=env.device) * 2.0 - 1.0
        obs, _, rews, dones, infos = env.step(actions)

        # --- capture frame ---
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
            os.makedirs(extra.video_dir, exist_ok=True)
            out_path = os.path.join(extra.video_dir, extra.video_name)
            imageio.mimsave(out_path, frames, fps=extra.fps, macro_block_size=None)
            print(f"Saved video: {out_path}  ({len(frames)} frames)")
        except ImportError:
            print("imageio not installed. Run: pip install imageio imageio-ffmpeg")
        finally:
            import shutil
            shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == '__main__':
    args = get_args()
    play(args)
