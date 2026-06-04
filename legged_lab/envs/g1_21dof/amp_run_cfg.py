# Copyright (c) 2021-2024, The RSL-RL Project Developers.

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.g1_21dof import G1_21DOF_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)


@configclass
class G1_21DOF_AMP_RunRewardCfg:
    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={"std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"std": math.sqrt(0.25)},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    waist_pos = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"]),
            "scale": 3.0,
            "tolerance": 0.0,
        },
    )

    leg_joint_position = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_ankle_roll_joint"]),
            "scale": 3.0,
        },
    )

    arm_pitch_position = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint", ".*_elbow_joint"]),
            "scale": 3.0,
        },
    )

    arm_roll_position = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_roll_joint"]),
            "scale": 3.0,
        },
    )

    arm_yaw_position = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_yaw_joint"]),
            "scale": 10.0,
        },
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    # base_height = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-10.0,
    #     params={"target_height": 0.78},
    # )

    # -- feet
    # feet_gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.5,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
    #     },
    # )
    
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=10.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
    #         "threshold": 0.5,
    #     },
    # )

    # 增加大迈步
    # feet_air_time_dense = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=1.25,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
    #         "threshold": 0.5,
    #     },
    # )
    
    # 增加交替接触
    # feet_contact = RewTerm(
    #     func=mdp.feet_contact_fixed,
    #     weight=0.25,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
    #         "stand_threshold": 0.1,
    #         "force_threshold": 5.0,
    #     },
    # )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
        },
    )
    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll_link"]),
    #     },
    # )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[".*knee_link", ".*shoulder_roll_link", ".*elbow_link", "pelvis", "waist_yaw_link"],
            ),
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)



@configclass
class G1_21DOF_AMP_RunFlatEnvCfg:
    device: str = "cuda:0"

    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=G1_21DOF_CFG,
        terrain_type="plane",
        terrain_generator=None,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )

    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=5,
        critic_obs_history_length=5,
        action_scale=0.25,
        terminate_contacts_body_names=[
            ".*knee_link", ".*shoulder_roll_link", ".*elbow_link", "pelvis", "waist_yaw_link",
        ],
        feet_body_names=[".*ankle_roll_link"],
    )

    reward = G1_21DOF_AMP_RunRewardCfg()

    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0, ang_vel=0.2, projected_gravity=1.0, commands=1.0,
            joint_pos=1.0, joint_vel=0.05, actions=1.0, height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )

    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,   # 10%的环境学习站立
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 3.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )

    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2, ang_vel=0.2, projected_gravity=0.05,
            joint_pos=0.01, joint_vel=1.5, height_scan=0.1,
        ),
    )

    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.3, 1.0),
                    "dynamic_friction_range": (0.3, 1.0),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (1.0, 1.0),
                    "velocity_range": (-1.0, 1.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(5.0, 5.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )

    sim: SimCfg = SimCfg(
        dt=0.005,
        decimation=4,
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )


@configclass
class G1_21DOF_AMP_RunAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "G1_21DOF_AmpOnPolicyRunner"
    experiment_name = "g1_21dof_amp_run"
    run_name = ""
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # --- AMP参数 ---
    amp_reward_coef = 1.0 #1.0
    amp_motion_files = [
        "legged_lab/datasets/g1_21dof/run_csv/C2-Run_to_stand_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C3-Run_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C4-Run_to_walk1_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C5-walk_to_run_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C6-stand_to_run_backwards_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C7-run_backwards_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C8-run_backwards_to_stand_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C9-run_backwards_turn_run_forward_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C10-run_backwards_stop_run_forward_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C11-run_turn_left90_poses.csv",
        "legged_lab/datasets/g1_21dof/run_csv/C14-run_turn_right90_poses.csv",
    ]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.9  #0.9
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 21
