from legged_gym.envs.base.legged_robot import LeggedRobot


class Go2(LeggedRobot):

    def check_termination(self):
        super().check_termination()
        # Terminate if the robot is flipped (projected gravity points upward)
        self.reset_buf |= self.projected_gravity[:, 2] > 0
