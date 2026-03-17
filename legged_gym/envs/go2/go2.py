import torch
from isaacgym import gymapi
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw


class Go2(LeggedRobot):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.debug_viz = True
        if self.viewer is not None:
            # Press D to toggle velocity arrow overlay
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "toggle_debug_viz")

    def render(self, sync_frame_time=True):
        if self.viewer:
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    import sys; sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_debug_viz" and evt.value > 0:
                    self.debug_viz = not self.debug_viz

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                if self.debug_viz:
                    self._draw_debug_vis()
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def check_termination(self):
        super().check_termination()
        # Terminate if the robot is flipped (projected gravity points upward)
        self.reset_buf |= self.projected_gravity[:, 2] > 0

    def _draw_debug_vis(self):
        """Draw commanded (blue) and actual (green) velocity arrows above each robot."""
        self.gym.clear_lines(self.viewer)

        arrow_scale = 0.5   # metres displayed per m/s
        viz_height  = 0.7   # offset above base origin [m]

        # Commanded velocity is in the body (heading) frame → rotate to world frame
        cmd_3d = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_3d[:, :2] = self.commands[:, :2]
        cmd_world = quat_apply_yaw(self.base_quat, cmd_3d)  # [N, 3] world frame

        # Actual linear velocity is already in world frame (root_states col 7-9)
        act_world = self.root_states[:, 7:9]  # [N, 2]

        base_pos = self.root_states[:, :3]    # [N, 3]

        cmd_world_np = cmd_world.cpu().numpy()
        act_world_np = act_world.cpu().numpy()
        base_pos_np  = base_pos.cpu().numpy()

        for i in range(self.num_envs):
            ox = base_pos_np[i, 0]
            oy = base_pos_np[i, 1]
            oz = base_pos_np[i, 2] + viz_height

            # Commanded velocity — blue
            cx = ox + cmd_world_np[i, 0] * arrow_scale
            cy = oy + cmd_world_np[i, 1] * arrow_scale
            self.gym.add_lines(self.viewer, self.envs[i], 1,
                               [ox, oy, oz, cx, cy, oz],
                               [0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

            # Actual velocity — green
            ax = ox + act_world_np[i, 0] * arrow_scale
            ay = oy + act_world_np[i, 1] * arrow_scale
            self.gym.add_lines(self.viewer, self.envs[i], 1,
                               [ox, oy, oz, ax, ay, oz],
                               [0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
