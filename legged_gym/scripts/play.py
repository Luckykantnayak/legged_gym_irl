# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # velocity tracking: plot for a single robot over one episode
    vel_tracking_log = defaultdict(list)
    # success metric: one episode across all envs
    eval_done = False
    total_steps = 0
    success_steps = 0

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

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
            # stop after one full episode length
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
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

    # print velocity tracking success rate (one episode, all envs)
    success_rate = success_steps / total_steps if total_steps > 0 else 0.0
    print(f"\n===== Velocity Tracking Evaluation (1 episode, {env.num_envs} envs) =====")
    print(f"Success rate (lin_err<0.2 m/s & ang_err<0.2 rad/s): {success_rate:.2%}")
    print(f"Total env-steps evaluated: {total_steps}")
    print(f"==========================================================\n")

    # plot velocity tracking for single robot
    fig_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'vel_tracking.pdf')
    _plot_velocity_tracking(vel_tracking_log, env.dt, robot_index, save_path=fig_path)

def _plot_velocity_tracking(log, dt, robot_index, save_path=None):
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
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved velocity tracking plot: {save_path}")
    plt.show()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
