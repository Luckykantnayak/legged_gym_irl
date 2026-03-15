from legged_gym.envs.base.legged_robot import LeggedRobot


class Go2(LeggedRobot):
    def check_termination(self):
        super().check_termination()
        flipped = self.projected_gravity[:, 2] > 0
        self.reset_buf |= flipped
