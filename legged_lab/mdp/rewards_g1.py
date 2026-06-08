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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import legged_lab.mdp as mdp

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.e1_21dof.e1_21dof_env import E1_21DOF_Env


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )

def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def joint_deviation_exp(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance: float = 0.1,
    scale: float = 3.0,
    max_err: float = 50.0,
) -> torch.Tensor:
    """Exp reward: 1.0 at default pos, decays to 0 as joints deviate (always active)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    default_pos = asset.data.default_joint_pos[:, joint_ids]
    joint_error = torch.norm(joint_pos - default_pos, dim=1)
    joint_error = torch.clamp(joint_error - tolerance, min=0.0, max=max_err)
    return torch.exp(-joint_error * scale)


def body_orientation_l2(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


# =========================================================
# feet Reward
# =========================================================
def feet_air_time_positive_biped(
    env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_air_time(
    env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps: on first contact after swing, reward (air_time - threshold).

    This is the primary step-driving reward — fires at every foot touchdown.
    Zero reward when velocity command is near zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_contact_fixed(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    stand_threshold: float = 0.06,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Reward valid foot contacts: standing → both feet on ground; walking → alternating single stance."""
    command = env.command_generator.command
    stand_cmd = torch.norm(command[:, :2], dim=1) < stand_threshold

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_history = contact_sensor.data.net_forces_w_history
    if contact_history is None:
        contact_history = contact_sensor.data.net_forces_w.unsqueeze(1)

    contacts = contact_history[:, :, sensor_cfg.body_ids, 2] > force_threshold
    contact_num = torch.sum(contacts, dim=-1)

    # Standing: reward if both feet on ground
    stand_contact = contact_num[:, -1] == 2
    reward = (stand_cmd & stand_contact).float()
    # Walking: reward if any timestep had exactly one foot in contact
    walk_contact = (~stand_cmd) & torch.any(contact_num == 1, dim=1)
    reward[walk_contact] = 1.0
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )

def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)

def feet_y_distance(env: BaseEnv) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# =========================================================
# Regularization Reward
# =========================================================
def ankle_torque(env: BaseEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: BaseEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: BaseEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)


def hip_yaw_action(env: BaseEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)

# ==========================================
# 腰部 (Waist Yaw) 惩罚
# ==========================================
def waist_yaw_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize waist yaw joint actions.
    惩罚腰部偏航关节的动作输出，防止躯干像拨浪鼓一样左右狂扭。
    """
    # 直接使用提前缓存好的 waist_ids
    return torch.sum(torch.abs(env.action[:, env.waist_ids]), dim=1)

def waist_yaw_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the waist yaw joint.
    惩罚腰部偏航关节输出过大的力矩，收紧核心，降低能耗。
    """
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.waist_ids]), dim=1)


# ==========================================
# 肩部俯仰 (Shoulder Pitch) 惩罚
# ==========================================
def shoulder_pitch_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize shoulder pitch joint actions.
    惩罚肩部俯仰关节的动作输出，限制手臂前后摆动的幅度。
    """
    # 索引 [0] 对应 left/right_shoulder_pitch_joint
    return torch.sum(torch.abs(env.action[:, [env.left_arm_ids[0], env.right_arm_ids[0]]]), dim=1)

def shoulder_pitch_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the shoulder pitch joints.
    惩罚肩部俯仰关节输出过大的力矩，防止极其生硬的机械甩臂。
    """
    return torch.sum(torch.square(env.robot.data.applied_torque[:, [env.left_arm_ids[0], env.right_arm_ids[0]]]), dim=1)

def shoulder_roll_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize shoulder roll joint actions.
    惩罚肩部横滚关节的动作输出，防止机器人走路时双臂过度向两侧“张开/侧平举”。
    """
    # 索引 [1] 对应 shoulder_roll_joint
    return torch.sum(torch.abs(env.action[:, [env.left_arm_ids[1], env.right_arm_ids[1]]]), dim=1)


def shoulder_roll_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the shoulder roll joints.
    惩罚肩部横滚关节输出过大的力矩，降低能耗。
    """
    return torch.sum(torch.square(env.robot.data.applied_torque[:, [env.left_arm_ids[1], env.right_arm_ids[1]]]), dim=1)


def shoulder_yaw_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize shoulder yaw joint actions.
    惩罚肩部偏航关节的动作输出，防止手臂绕垂直轴过度“内旋或外旋”。
    """
    # 索引 [2] 对应 shoulder_yaw_joint
    return torch.sum(torch.abs(env.action[:, [env.left_arm_ids[2], env.right_arm_ids[2]]]), dim=1)


def shoulder_yaw_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the shoulder yaw joints.
    惩罚肩部偏航关节输出过大的力矩。
    """
    return torch.sum(torch.square(env.robot.data.applied_torque[:, [env.left_arm_ids[2], env.right_arm_ids[2]]]), dim=1)

def elbow_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize elbow joint actions.
    惩罚肘部关节的动作输出，防止手臂像大风车一样疯狂甩动。
    """
    # 使用 L1 范数 (abs)，对任何微小的乱动都保持敏感
    return torch.sum(torch.abs(env.action[:, [env.left_arm_ids[3], env.right_arm_ids[3]]]), dim=1)


def elbow_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the elbow joints.
    惩罚肘部关节输出过大的力矩，降低能耗，让手臂动作更轻柔。
    """
    # 使用 L2 范数 (square)，重点打击极端的爆发性发力
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.elbow_joint_ids]), dim=1)


# =========================================================
# Periodic gait-based reward function
# =========================================================
def gait_clock(phase, air_ratio, delta_t):
    # (保持原有逻辑完全不变，仅做排版)
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd

# 摆动相脚底不受力奖励，鼓励机器人脚底不受力
def gait_feet_frc_perio(env: BaseEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score

# 支撑相脚底无速度奖励，鼓励机器人脚底不动
def gait_feet_spd_perio(env: BaseEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score

# 支撑相真实承重奖励，鼓励机器人该踩地的时候把真实的体重，不能轻轻点着
def gait_feet_frc_support_perio(env: BaseEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_generator.command
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# ---- Gait periodic rewards ----

def feet_gait(
    env: BaseEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward periodic L/R alternating foot contact based on episode time.

    Uses a global phase clock (no phase observations needed).
    Rewards when foot contact matches the expected stance/swing phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = [(global_phase + o) % 1.0 for o in offset]
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_generator.command[:, :2], dim=1)
        reward *= cmd_norm > 0.1
    return reward


def foot_clearance_reward(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing a specified height off the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def air_time_variance_penalty(
    env: BaseEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize variance in air/contact time between left and right feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return (
        torch.var(torch.clip(last_air_time, max=0.5), dim=1)
        + torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    )



# ---- Gait periodic rewards (from Unitree RL Lab) ----

def feet_gait(
    env: BaseEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward periodic L/R alternating foot contact based on episode time.

    Uses a global phase clock (no phase observations needed).
    Penalizes when foot contact doesn't match the expected stance phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = [(global_phase + o) % 1.0 for o in offset]
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_generator.command[:, :2], dim=1)
        reward *= cmd_norm > 0.1
    return reward


def foot_clearance_reward(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing a specified height off the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def air_time_variance_penalty(
    env: BaseEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize variance in air/contact time between left and right feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return (
        torch.var(torch.clip(last_air_time, max=0.5), dim=1)
        + torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    )


# ---- Unitree port: additional reward functions ----

def is_alive(env: BaseEnv) -> torch.Tensor:
    """Reward for staying alive (not terminated)."""
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)


def joint_vel_l2(env: BaseEnv) -> torch.Tensor:
    """Penalize joint velocities."""
    asset: Articulation = env.scene["robot"]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def base_height_l2(env: BaseEnv, target_height: float = 0.78) -> torch.Tensor:
    """Penalize deviation of base height from target."""
    asset: Articulation = env.scene["robot"]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


# =========================================================
# engineai_amp reward functions (ported from engineai_lab)
# =========================================================

def _epoch_curriculum_scale(env: BaseEnv, start_scale: float, power: float, interval_epochs: int) -> float:
    """Compute epoch-based scale by exponentiating by power every interval_epochs."""
    num_step = env.common_step_counter if hasattr(env, "common_step_counter") else env.sim_step_counter
    interval = max(int(interval_epochs), 1)
    updates = num_step // interval
    return float(start_scale) ** (float(power) ** updates)


def action_smoothness(env: BaseEnv) -> torch.Tensor:
    """Penalize action second-order differences to encourage smooth control."""
    buf = env.action_buffer._circular_buffer.buffer
    prev_action = buf[:, -1, :]
    prev_prev_action = buf[:, -2, :]
    prev_prev_prev_action = buf[:, -3, :] if buf.shape[1] >= 3 else prev_prev_action
    second_diff = prev_action + prev_prev_prev_action - 2.0 * prev_prev_action
    reward = torch.sum(torch.square(second_diff), dim=1)
    if hasattr(env, "reset_env_ids") and env.reset_env_ids.numel() > 0:
        reward[env.reset_env_ids] = 0.0
    return reward


def action_smoothness_with_curriculum(
    env: BaseEnv, start_scale: float, power: float, interval_epochs: int
) -> torch.Tensor:
    """Action smoothness penalty with epoch-based curriculum scaling."""
    return action_smoothness(env) * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def action_rate_with_curriculum(
    env: BaseEnv, start_scale: float, power: float, interval_epochs: int
) -> torch.Tensor:
    """Action rate penalty with epoch-based curriculum scaling."""
    return action_rate_l2(env) * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def energy_cost_ea(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize energy consumption approximated by the sum of squared joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_torques = asset.data.applied_torque[:, :]
    joint_vel = asset.data.joint_vel[:, :]
    power = joint_torques * joint_vel
    return torch.sum(torch.abs(power), dim=1)


def energy_cost_with_curriculum(
    env: BaseEnv,
    start_scale: float,
    power: float,
    interval_epochs: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Energy cost penalty with epoch-based curriculum scaling."""
    return energy_cost_ea(env, asset_cfg=asset_cfg) * _epoch_curriculum_scale(
        env, start_scale, power, interval_epochs
    )


def base_orientation_ea(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward keeping the base roll/pitch near zero (exp kernel)."""
    from isaaclab.utils.math import euler_xyz_from_quat
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    base_euler = torch.stack((roll, pitch), dim=-1)
    return torch.exp(-torch.sum(torch.abs(base_euler), dim=-1) * 10.0)


def base_height_tracking_ea(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = 0.82
) -> torch.Tensor:
    """Reward keeping the base height near a target height (exp kernel)."""
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.abs(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error * 30.0)


def feet_position_ea(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,
    stand_threshold: float = 0.06,
    ankle_distance: float = 0.22,
    base_height_target: float = 0.82,
) -> torch.Tensor:
    """Reward keeping feet near desired stance when standing; otherwise return 1."""
    from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_apply_inverse
    command = env.command_generator.command
    stand_command = (torch.norm(command[:, :2], dim=1) < stand_threshold) & (
        torch.abs(command[:, 2]) < stand_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w

    r, p, y = euler_xyz_from_quat(base_quat_w)
    heading_quat = quat_from_euler_xyz(torch.zeros_like(r), torch.zeros_like(p), y)
    feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
    num_envs, num_feet, _ = feet_pos_rel.shape
    heading_quat_per_foot = heading_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    feet_pos_rel_flat = feet_pos_rel.reshape(-1, 3)
    feet_pos_heading = quat_apply_inverse(heading_quat_per_foot, feet_pos_rel_flat).reshape(num_envs, num_feet, 3)

    desired_x = torch.zeros((num_envs, num_feet), device=feet_pos_heading.device)
    desired_y = torch.cat(
        (
            (ankle_distance * 0.5) * torch.ones((num_envs, num_feet // 2), device=feet_pos_heading.device),
            (-ankle_distance * 0.5) * torch.ones((num_envs, num_feet - num_feet // 2), device=feet_pos_heading.device),
        ),
        dim=1,
    )
    desired_z = -(base_height_target - 0.045) * torch.ones((num_envs, num_feet), device=feet_pos_heading.device)
    desired = torch.stack((desired_x, desired_y, desired_z), dim=-1)

    position_error = torch.sum(torch.abs(feet_pos_heading - desired), dim=(1, 2))
    reward_stand = torch.exp(-position_error * 3.0)
    return torch.where(stand_command, reward_stand, torch.ones_like(reward_stand))


def feet_orientation_ea(
    env: BaseEnv, asset_cfg: SceneEntityCfg, stand_threshold: float = 0.06
) -> torch.Tensor:
    """Reward aligning feet orientation; ignore yaw error while turning."""
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
    command = env.command_generator.command
    yaw_command = torch.abs(command[:, 2]) > stand_threshold

    asset: Articulation = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    base_quat = asset.data.root_quat_w

    num_envs, num_feet, _ = feet_quat.shape
    feet_flat = feet_quat.reshape(-1, 4)
    roll, pitch, yaw = euler_xyz_from_quat(feet_flat)
    roll = roll.reshape(num_envs, num_feet)
    pitch = pitch.reshape(num_envs, num_feet)
    yaw = yaw.reshape(num_envs, num_feet)

    _, _, base_yaw = euler_xyz_from_quat(base_quat)
    feet_roll_pitch_error = torch.sum(torch.abs(torch.stack((roll, pitch), dim=-1)), dim=-1)
    feet_yaw_error = torch.abs(wrap_to_pi(yaw - base_yaw.unsqueeze(1)))

    rew = torch.sum(feet_roll_pitch_error + feet_yaw_error, dim=1)
    rew[yaw_command] = torch.sum(feet_roll_pitch_error[yaw_command], dim=1)
    return torch.exp(-rew * 2.0)


def feet_stumble_ea(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, tangential_threshold: float = 2.0, normal_threshold: float = 1.0
) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces using contact forces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    tangential = torch.norm(forces[..., :2], dim=-1) > tangential_threshold
    small_normal = torch.abs(forces[..., 2]) < normal_threshold
    stumble = tangential & small_normal
    return stumble.sum(dim=1)


def feet_air_time_similarity(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = 4.0,
    min_air_time: float = 0.0,
) -> torch.Tensor:
    """Reward similar air time between two feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if body_ids is None or len(body_ids) != 2:
        raise ValueError("feet_air_time_similarity expects exactly two foot body ids in sensor_cfg.body_ids.")
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, body_ids]
    recent_contact = torch.any(first_contact > 0.0, dim=1)
    valid = torch.all(last_air_time > min_air_time, dim=1)
    diff = torch.abs(last_air_time[:, 0] - last_air_time[:, 1])
    reward = torch.exp(-diff * scale)
    return reward * (recent_contact & valid)
