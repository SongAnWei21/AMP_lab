# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Configuration for Unitree G1 21DOF humanoid robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

# G1 rotor inertia (kg*m^2) -- unified across all joints
G1_ARMATURE = 0.01

# G1 actuator limits from URDF

# Legs
G1_HIP_PITCH_EFFORT = 88
G1_HIP_PITCH_VELOCITY = 32
G1_HIP_ROLL_EFFORT = 139
G1_HIP_ROLL_VELOCITY = 20
G1_HIP_YAW_EFFORT = 88
G1_HIP_YAW_VELOCITY = 32
G1_KNEE_EFFORT = 139
G1_KNEE_VELOCITY = 20

# Feet
G1_ANKLE_PITCH_EFFORT = 50
G1_ANKLE_PITCH_VELOCITY = 37
G1_ANKLE_ROLL_EFFORT = 50
G1_ANKLE_ROLL_VELOCITY = 37

# Waist
G1_WAIST_YAW_EFFORT = 88
G1_WAIST_YAW_VELOCITY = 32

# Arms
G1_SHOULDER_PITCH_EFFORT = 25
G1_SHOULDER_PITCH_VELOCITY = 37
G1_SHOULDER_ROLL_EFFORT = 25
G1_SHOULDER_ROLL_VELOCITY = 37
G1_SHOULDER_YAW_EFFORT = 25
G1_SHOULDER_YAW_VELOCITY = 37
G1_ELBOW_EFFORT = 25
G1_ELBOW_VELOCITY = 37

# G1 21DOF robot model configuration
G1_21DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAAC_ASSET_DIR}/g1_21dof/g1_21dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.15,
            "left_hip_roll_joint": 0,
            "left_hip_yaw_joint": 0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0,

            "right_hip_pitch_joint": -0.15,
            "right_hip_roll_joint": 0,
            "right_hip_yaw_joint": 0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0,

            "waist_yaw_joint": 0.0,

            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.25,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.0,

            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": G1_HIP_PITCH_EFFORT,
                ".*_hip_roll_joint": G1_HIP_ROLL_EFFORT,
                ".*_hip_yaw_joint": G1_HIP_YAW_EFFORT,
                ".*_knee_joint": G1_KNEE_EFFORT,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": G1_HIP_PITCH_VELOCITY,
                ".*_hip_roll_joint": G1_HIP_ROLL_VELOCITY,
                ".*_hip_yaw_joint": G1_HIP_YAW_VELOCITY,
                ".*_knee_joint": G1_KNEE_VELOCITY,
            },
            stiffness={
                ".*_hip_pitch_joint": 100,
                ".*_hip_roll_joint": 100,
                ".*_hip_yaw_joint": 50,
                ".*_knee_joint": 100,
            },
            damping={
                ".*_hip_pitch_joint": 4,
                ".*_hip_roll_joint": 4,
                ".*_hip_yaw_joint": 2.5,
                ".*_knee_joint": 4,
            },
            armature={
                ".*_hip_pitch_joint": G1_ARMATURE,
                ".*_hip_roll_joint": G1_ARMATURE,
                ".*_hip_yaw_joint": G1_ARMATURE,
                ".*_knee_joint": G1_ARMATURE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint"
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": G1_ANKLE_PITCH_EFFORT,
                ".*_ankle_roll_joint": G1_ANKLE_ROLL_EFFORT,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": G1_ANKLE_PITCH_VELOCITY,
                ".*_ankle_roll_joint": G1_ANKLE_ROLL_VELOCITY,
            },
            stiffness={
                ".*_ankle_pitch_joint": 20,
                ".*_ankle_roll_joint": 20,
            },
            damping={
                ".*_ankle_pitch_joint": 2,
                ".*_ankle_roll_joint": 2,
            },
            armature={
                ".*_ankle_pitch_joint": G1_ARMATURE,
                ".*_ankle_roll_joint": G1_ARMATURE,
            },
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=G1_WAIST_YAW_EFFORT,
            velocity_limit_sim=G1_WAIST_YAW_VELOCITY,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=100,
            damping=4,
            armature=G1_ARMATURE,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": G1_SHOULDER_PITCH_EFFORT,
                ".*_shoulder_roll_joint": G1_SHOULDER_ROLL_EFFORT,
                ".*_shoulder_yaw_joint": G1_SHOULDER_YAW_EFFORT,
                ".*_elbow_joint": G1_ELBOW_EFFORT,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": G1_SHOULDER_PITCH_VELOCITY,
                ".*_shoulder_roll_joint": G1_SHOULDER_ROLL_VELOCITY,
                ".*_shoulder_yaw_joint": G1_SHOULDER_YAW_VELOCITY,
                ".*_elbow_joint": G1_ELBOW_VELOCITY,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 30,
                ".*_shoulder_roll_joint": 30,
                ".*_shoulder_yaw_joint": 30,
                ".*_elbow_joint": 30,
            },
            damping={
                ".*_shoulder_pitch_joint": 2,
                ".*_shoulder_roll_joint": 2,
                ".*_shoulder_yaw_joint": 2,
                ".*_elbow_joint": 2,
            },
            armature={
                ".*_shoulder_pitch_joint": G1_ARMATURE,
                ".*_shoulder_roll_joint": G1_ARMATURE,
                ".*_shoulder_yaw_joint": G1_ARMATURE,
                ".*_elbow_joint": G1_ARMATURE,
            },
        ),
    },
)
