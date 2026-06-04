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

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

# 灵足各型号电机的转动惯量参数 kg*m^2
ARMATURE_RS00 = 0.001   # 14Nm  
ARMATURE_RS01 = 0.0042  
ARMATURE_RS02 = 0.0042
ARMATURE_RS03 = 0.02   # 60Nm
ARMATURE_RS04 = 0.04
ARMATURE_RS05 = 0.0007
ARMATURE_RS06 = 0.012  # 36Nm

EFFORT_LIMIT_RS00 = 14
VELOCITY_LIMIT_RS00 = 32.99
EFFORT_LIMIT_RS03 = 60
VELOCITY_LIMIT_RS03 = 20.42
EFFORT_LIMIT_RS06 = 36
VELOCITY_LIMIT_RS06 = 50.27
EFFORT_LIMIT_RS06_17_28 = 59.3  # RS06同步带传动减速比17/28 
VELOCITY_LIMIT_RS06_17_28 = 30.52 # RS06同步带传动减速比17/28 

# E1机器人模型配置
E1_21DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAAC_ASSET_DIR}/e1_21dof/urdf/E1_21dof.urdf", 
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

            "left_hip_pitch_joint": -0.2,
            "left_hip_roll_joint": 0,
            "left_hip_yaw_joint": 0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.1,

            "right_hip_pitch_joint": -0.2,
            "right_hip_roll_joint": 0,
            "right_hip_yaw_joint": 0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.1,

            "waist_yaw_joint":0.0,

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
                ".*_hip_pitch_joint": EFFORT_LIMIT_RS03,
                ".*_hip_roll_joint": EFFORT_LIMIT_RS03,
                ".*_hip_yaw_joint": EFFORT_LIMIT_RS06,
                ".*_knee_joint": EFFORT_LIMIT_RS06_17_28,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": VELOCITY_LIMIT_RS03,
                ".*_hip_roll_joint": VELOCITY_LIMIT_RS03,
                ".*_hip_yaw_joint": VELOCITY_LIMIT_RS06,
                ".*_knee_joint": VELOCITY_LIMIT_RS06_17_28,
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
                ".*_hip_pitch_joint":  ARMATURE_RS03,
                ".*_hip_roll_joint": ARMATURE_RS03,
                ".*_hip_yaw_joint": ARMATURE_RS06,
                ".*_knee_joint": ARMATURE_RS06,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint", 
                ".*_ankle_roll_joint"
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": EFFORT_LIMIT_RS06_17_28,
                ".*_ankle_roll_joint": EFFORT_LIMIT_RS00,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": VELOCITY_LIMIT_RS06_17_28,
                ".*_ankle_roll_joint": VELOCITY_LIMIT_RS00,
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
                ".*_ankle_pitch_joint": ARMATURE_RS06,
                ".*_ankle_roll_joint": ARMATURE_RS00,
            },
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=EFFORT_LIMIT_RS03,
            velocity_limit_sim=VELOCITY_LIMIT_RS03,
            joint_names_expr=["waist_yaw_joint"], 
            stiffness=100,
            damping=4, 
            armature=ARMATURE_RS03,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": EFFORT_LIMIT_RS06,
                ".*_shoulder_roll_joint": EFFORT_LIMIT_RS06,
                ".*_shoulder_yaw_joint": EFFORT_LIMIT_RS00,
                ".*_elbow_joint": EFFORT_LIMIT_RS06,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": VELOCITY_LIMIT_RS06,
                ".*_shoulder_roll_joint": VELOCITY_LIMIT_RS06,
                ".*_shoulder_yaw_joint": VELOCITY_LIMIT_RS00,
                ".*_elbow_joint": VELOCITY_LIMIT_RS06,
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
                ".*_shoulder_pitch_joint": ARMATURE_RS06,
                ".*_shoulder_roll_joint": ARMATURE_RS06,
                ".*_shoulder_yaw_joint": ARMATURE_RS00,
                ".*_elbow_joint": ARMATURE_RS06,
            },
        ),
    },
)