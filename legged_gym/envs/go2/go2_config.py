from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.base_config import BaseConfig

class GO2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'


class GO2FlatCfg( GO2RoughCfg ):
    class env( GO2RoughCfg.env ):
        num_envs = 4096
        num_observations = 48

    class terrain( GO2RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class asset( GO2RoughCfg.asset ):
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards( GO2RoughCfg.rewards ):
        max_contact_force = 350.
        class scales( GO2RoughCfg.rewards.scales ):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.
            # feet_contact_forces = -0.01

    class commands( GO2RoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( GO2RoughCfg.commands.ranges ):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand( GO2RoughCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

class GO2FlatCfgPPO( GO2RoughCfgPPO ):
    class policy( GO2RoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( GO2RoughCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( GO2RoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_go2'


class GO2FlatCfgSAC( BaseConfig ):
    seed = 1

    class policy:
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm:
        # action bounds: SAC applies tanh internally, env scales actions by action_scale=0.25
        action_max = 1.0
        action_min = -1.0
        # learning rates
        actor_lr = 3e-4
        critic_lr = 3e-4
        alpha = 0.2       # initial entropy coefficient
        alpha_lr = 3e-4
        # replay buffer
        storage_size = 1_000_000
        storage_initial_size = 10_000  # transitions before first update
        # mini-batch training
        batch_size = 256
        batch_count = 4   # gradient steps per data collection step
        # discount
        gamma = 0.99
        # n-step returns (1 = standard TD(0), 3-5 improves long-horizon credit assignment)
        n_step_returns = 3
        # misc
        gradient_clip = 1.0

    class runner:
        algorithm_class_name = 'SAC'
        num_steps_per_env = 1   # collect 1 env step per iteration (off-policy)
        max_iterations = 15_000
        # logging
        save_interval = 500
        experiment_name = 'flat_go2_sac'
        run_name = ''
        # resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None


class GO2FlatCfgDDPG( BaseConfig ):
    seed = 1

    class policy:
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm:
        # learning rates
        actor_lr = 1e-4
        critic_lr = 1e-3
        # exploration: Gaussian noise std added to deterministic actions in [-1, 1]
        action_noise_std = 0.1
        # replay buffer
        storage_size = 1_000_000
        storage_initial_size = 10_000  # transitions before first update
        # mini-batch training
        batch_size = 256
        batch_count = 4   # gradient steps per data collection step
        # discount & target network
        gamma = 0.99
        polyak = 0.995
        # n-step returns (1 = standard TD(0), 3-5 improves long-horizon credit assignment)
        n_step_returns = 3
        # misc
        gradient_clip = 1.0

    class runner:
        algorithm_class_name = 'DDPG'
        num_steps_per_env = 1   # collect 1 env step per iteration (off-policy)
        max_iterations = 15_000
        # logging
        save_interval = 500
        experiment_name = 'flat_go2_ddpg'
        run_name = ''
        # resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
