import argparse
import glob
import os
import sys
import tempfile
from collections import defaultdict

import isaacgym
import matplotlib.pyplot as plt
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
_rec_parser.add_argument("--cmd_seed", type=int, default=None,
                         help="Fix RNG seed before rollout so command resampling is identical across runs")
_rec_parser.add_argument("--cmd_seq", type=float, nargs="+", default=None,
                         help="Hardcoded command sequence as flat list of floats (groups of 3: vx vy vyaw). "
                              "E.g. --cmd_seq 0.5 0.0 0.0  1.0 0.0 0.5")
# evaluation mode
_rec_parser.add_argument("--evaluate", action="store_true",
                         help="Run policy evaluation (mean/std/SEM of return over iterations) instead of interactive play")
_rec_parser.add_argument("--num_iters", type=int, default=100,
                         help="Number of evaluation iterations (only used with --evaluate)")
_rec_parser.add_argument("--eval_seed", type=int, default=None,
                         help="Random seed for evaluation rollouts (only used with --evaluate)")
_rec_parser.add_argument("--eval_output", type=str, default="eval_policy_returns.png",
                         help="Plot output path for evaluation results")
_rec_parser.add_argument("--eval_log_path", type=str, default="eval_policy_returns.npz",
                         help="Path to save evaluation arrays (npz)")
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

    # velocity tracking: plot for a single robot over one episode
    vel_tracking_log = defaultdict(list)
    # success metric: one episode across all envs
    eval_done = False
    total_steps = 0
    success_steps = 0

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

    # --- fixed command sequence ---
    if rec_args.cmd_seq is not None:
        raw = rec_args.cmd_seq
        assert len(raw) % 3 == 0, "--cmd_seq must have a multiple-of-3 number of values"
        cmd_list = [[raw[i], raw[i + 1], raw[i + 2]] for i in range(0, len(raw), 3)]
        _call_count = [0]

        def _seq_resample(env_ids, init=False):
            vx, vy, vyaw = cmd_list[_call_count[0] % len(cmd_list)]
            env.commands[env_ids, 0] = vx
            env.commands[env_ids, 1] = vy
            env.commands[env_ids, 2] = vyaw
            if len(env_ids) > 0 and not init:
                _call_count[0] += 1

        env._resample_commands = _seq_resample
        env._resample_commands(torch.arange(env.num_envs, device=env.device), init=True)
    # --- fix command RNG seed if requested ---
    elif rec_args.cmd_seed is not None:
        torch.manual_seed(rec_args.cmd_seed)
        np.random.seed(rec_args.cmd_seed)
        env._resample_commands(torch.arange(env.num_envs, device=env.device))

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

        # velocity tracking plot: single robot over first episode
        if i < int(env.max_episode_length):
            vel_tracking_log['cmd_x'].append(env.commands[robot_index, 0].item())
            vel_tracking_log['cmd_y'].append(env.commands[robot_index, 1].item())
            vel_tracking_log['cmd_yaw'].append(env.commands[robot_index, 2].item())
            vel_tracking_log['vel_x'].append(env.base_lin_vel[robot_index, 0].item())
            vel_tracking_log['vel_y'].append(env.base_lin_vel[robot_index, 1].item())
            vel_tracking_log['vel_yaw'].append(env.base_ang_vel[robot_index, 2].item())

        # success metric: accumulate over first episode across all envs
        if not eval_done:
            cmd_xy = env.commands[:, :2]
            vel_xy = env.base_lin_vel[:, :2]
            lin_err = torch.sqrt(torch.sum(torch.square(cmd_xy - vel_xy), dim=1))
            ang_err = torch.abs(env.commands[:, 2] - env.base_ang_vel[:, 2])
            success_steps += ((lin_err < 0.2) & (ang_err < 0.2)).sum().item()
            total_steps += env.num_envs
            if i + 1 >= int(env.max_episode_length):
                eval_done = True

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

    # --- velocity tracking success rate (one episode, all envs) ---
    success_rate = success_steps / total_steps if total_steps > 0 else 0.0
    print(f"\n===== Velocity Tracking Evaluation (1 episode, {env.num_envs} envs) =====")
    print(f"Success rate (lin_err<0.2 m/s & ang_err<0.2 rad/s): {success_rate:.2%}")
    print(f"Total env-steps evaluated: {total_steps}")
    print(f"==========================================================\n")

    # plot velocity tracking for single robot
    _plot_velocity_tracking(vel_tracking_log, env.dt, robot_index)

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


def _plot_velocity_tracking(log, dt, robot_index):
    """Plot commanded vs actual velocities for a single robot over one episode."""
    n = len(log['cmd_x'])
    time = np.arange(n) * dt

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(time, log['vel_x'], label='actual', alpha=0.8)
    axs[0].plot(time, log['cmd_x'], label='commanded', linestyle='--', alpha=0.8)
    axs[0].set_ylabel('Lin vel x [m/s]')
    axs[0].set_title(f'Velocity Tracking (robot {robot_index}, 1 episode)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(time, log['vel_y'], label='actual', alpha=0.8)
    axs[1].plot(time, log['cmd_y'], label='commanded', linestyle='--', alpha=0.8)
    axs[1].set_ylabel('Lin vel y [m/s]')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(time, log['vel_yaw'], label='actual', alpha=0.8)
    axs[2].plot(time, log['cmd_yaw'], label='commanded', linestyle='--', alpha=0.8)
    axs[2].set_ylabel('Ang vel yaw [rad/s]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = get_args()
    play(args)
