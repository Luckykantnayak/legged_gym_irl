from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import glob
import shutil
import tempfile
import argparse
from collections import defaultdict

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import matplotlib.pyplot as plt


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--cmd_seed", type=int, default=None,
                     help="Fix RNG seed before rollout so command resampling is identical across runs")
_parser.add_argument("--cmd_seq", type=float, nargs="+", default=None,
                     help="Hardcoded command sequence as flat list of floats (groups of 3: vx vy vyaw). "
                          "E.g. --cmd_seq 0.5 0.0 0.0  1.0 0.0 0.5")
_parser.add_argument("--record", action="store_true",
                     help="Record a video of the rollout via an Isaac Gym camera sensor.")
_parser.add_argument("--video_dir", type=str, default="videos/")
_parser.add_argument("--video_name", type=str, default="episode.mp4")
_parser.add_argument("--fps", type=int, default=30)
_parser.add_argument("--video_steps", type=int, default=-1,
                     help="Number of rollout steps to record; -1 uses env.max_episode_length.")
_parser.add_argument("--cam_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                     help="Override viewer/recording camera position in world coords.")
_parser.add_argument("--cam_lookat", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                     help="Override viewer/recording camera look-at target in world coords.")
_parser.add_argument("--log_attention", action="store_true",
                     help="Record per-step attention weights (env 0, last transformer block) and plot a "
                          "time × context-position heatmap averaged over heads.")
_parser.add_argument("--attention_env", type=int, default=0,
                     help="Env index whose attention is logged when --log_attention is set.")
_extra_args, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining


def play(args):
    # Viewer-based capture (matches play_off_policy.py) requires a display.
    if _extra_args.record:
        args.headless = False

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Apply cam overrides before env construction so the viewer is created at the right pose.
    if _extra_args.cam_pos is not None:
        env_cfg.viewer.pos = list(_extra_args.cam_pos)
    if _extra_args.cam_lookat is not None:
        env_cfg.viewer.lookat = list(_extra_args.cam_lookat)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if env.viewer is not None and (_extra_args.cam_pos is not None or _extra_args.cam_lookat is not None):
        env.gym.viewer_camera_look_at(
            env.viewer, None,
            gymapi.Vec3(*env_cfg.viewer.pos),
            gymapi.Vec3(*env_cfg.viewer.lookat),
        )

    if _extra_args.cmd_seq is not None:
        raw = _extra_args.cmd_seq
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
    elif _extra_args.cmd_seed is not None:
        torch.manual_seed(_extra_args.cmd_seed)
        np.random.seed(_extra_args.cmd_seed)
        env._resample_commands(torch.arange(env.num_envs, device=env.device))

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    actor_critic = ppo_runner.alg.actor_critic
    policy = ppo_runner.get_inference_policy(device=env.device)
    # Flush any stale rolling context from training before first rollout step.
    actor_critic.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    vel_tracking_log = defaultdict(list)
    eval_done = False
    total_steps = 0
    success_steps = 0

    # Attention logging: grab the actor transformer's last block each step. Each `last_attn`
    # is (B, heads, S, S) with S = context_length + 1 at rollout; we keep the final query row.
    attn_log = []
    attn_block = None
    if _extra_args.log_attention:
        attn_block = actor_critic.memory_a.transformer.blocks[-1].sa

    # Video recording reuses the viewer (write_viewer_image_to_file), matching play_off_policy.py.
    frame_dir = None
    frame_idx = 0
    record_limit = int(env.max_episode_length) if _extra_args.video_steps < 0 else _extra_args.video_steps
    if _extra_args.record:
        if env.viewer is None:
            print("WARNING: viewer is None - cannot record. Re-run without --headless or with a display.")
        else:
            frame_dir = tempfile.mkdtemp(prefix="legged_frames_")
            print(f"Recording frames to tmp dir: {frame_dir}")
            print(f"Output video: {os.path.join(_extra_args.video_dir, _extra_args.video_name)}")

    for i in range(int(env.max_episode_length)):
        actions = policy(obs.detach())

        # Capture right after the forward pass; the next policy() call will overwrite last_attn.
        if attn_block is not None and attn_block.last_attn is not None and i < int(env.max_episode_length):
            # last_attn: (B, heads, S, S). Row -1 = current query's attention over all keys.
            row = attn_block.last_attn[_extra_args.attention_env, :, -1, :].cpu().numpy()
            attn_log.append(row)

        obs, _, rews, dones, infos = env.step(actions.detach())
        # Transformer carries a rolling context per env; zero it for envs that just terminated.
        actor_critic.reset(dones)

        if RECORD_FRAMES and i % 2:
            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name,
                                    'exported', 'frames', f"{img_idx}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if frame_dir is not None and env.viewer is not None and i < record_limit:
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
            env.gym.write_viewer_image_to_file(env.viewer, frame_path)
            frame_idx += 1

        if i < int(env.max_episode_length):
            vel_tracking_log['cmd_x'].append(env.commands[robot_index, 0].item())
            vel_tracking_log['cmd_y'].append(env.commands[robot_index, 1].item())
            vel_tracking_log['cmd_yaw'].append(env.commands[robot_index, 2].item())
            vel_tracking_log['vel_x'].append(env.base_lin_vel[robot_index, 0].item())
            vel_tracking_log['vel_y'].append(env.base_lin_vel[robot_index, 1].item())
            vel_tracking_log['vel_yaw'].append(env.base_ang_vel[robot_index, 2].item())

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

    success_rate = success_steps / total_steps if total_steps > 0 else 0.0
    print(f"\n===== Velocity Tracking Evaluation (1 episode, {env.num_envs} envs) =====")
    print(f"Success rate (lin_err<0.2 m/s & ang_err<0.2 rad/s): {success_rate:.2%}")
    print(f"Total env-steps evaluated: {total_steps}")
    print(f"==========================================================\n")

    if frame_dir is not None and frame_idx > 0:
        try:
            import imageio
            png_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
            frames = [imageio.imread(p) for p in png_files]
            os.makedirs(_extra_args.video_dir, exist_ok=True)
            out_path = os.path.join(_extra_args.video_dir, _extra_args.video_name)
            imageio.mimsave(out_path, frames, fps=_extra_args.fps, macro_block_size=None)
            print(f"[play_transformer] saved video: {out_path} ({len(frames)} frames)")
        except ImportError:
            print("[play_transformer] imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    if attn_log:
        os.makedirs(_extra_args.video_dir, exist_ok=True)
        stem = os.path.splitext(_extra_args.video_name)[0]
        npz_path = os.path.join(_extra_args.video_dir, f"{stem}_attn.npz")
        png_path = os.path.join(_extra_args.video_dir, f"{stem}_attn.png")
        data = np.stack(attn_log, axis=0)  # (T, heads, S)
        np.savez(npz_path, attn=data, dt=env.dt, env_index=_extra_args.attention_env)
        print(f"[play_transformer] saved attention log: {npz_path} shape={data.shape}")
        _plot_attention(attn_log, env.dt, _extra_args.attention_env, save_path=png_path)

    _plot_velocity_tracking(vel_tracking_log, env.dt, robot_index)


def _plot_attention(attn_rows, dt, env_index, save_path=None):
    """Plot (T × context_position) attention heatmap averaged over heads, plus per-head panels.

    attn_rows: list of (heads, S) arrays where S = context_length + 1. Column index maps
    from oldest context token (0) to current token (S-1).
    """
    data = np.stack(attn_rows, axis=0)  # (T, heads, S)
    T, heads, S = data.shape
    time = np.arange(T) * dt
    # Column labels: -(S-1) for oldest context, 0 for current token.
    offsets = np.arange(S) - (S - 1)

    fig, axs = plt.subplots(1, heads + 1, figsize=(4 * (heads + 1), 4), sharey=True)
    if heads + 1 == 1:
        axs = [axs]

    def _imshow(ax, mat, title):
        im = ax.imshow(mat.T, aspect='auto', origin='lower',
                       extent=[time[0], time[-1], offsets[0] - 0.5, offsets[-1] + 0.5],
                       cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel('Time [s]')
        return im

    im = _imshow(axs[0], data.mean(axis=1), f'mean over heads')
    axs[0].set_ylabel('context offset (0 = current)')
    for h in range(heads):
        _imshow(axs[h + 1], data[:, h, :], f'head {h}')

    fig.colorbar(im, ax=axs, shrink=0.8, label='attention weight')
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[play_transformer] saved attention heatmap: {save_path}")
    plt.show()


def _plot_velocity_tracking(log, dt, robot_index):
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
    # JIT export only supports LSTM memory today; transformer memory has no `rnn` attribute.
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
