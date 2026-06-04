# AMP Lab — G1 21-DOF 人形机器人 AMP 强化学习训练框架

基于 [Isaac Lab](https://isaaclab.github.io/) + [RSL-RL](https://github.com/leggedrobotics/rsl_rl)，专门训练 Unitree G1 21 自由度人形机器人的行走和跑步策略，集成 AMP（对抗运动先验）和纯 RL 两种训练方式。

## 环境

依赖 Isaac Lab 2.3.2 和相应的 conda 环境：

```bash
conda activate amp_lab
```

## 快速开始

### 训练

```bash
# AMP 行走训练
python legged_lab/scripts/train.py --task=g1_21dof_amp_walk --headless

# AMP 跑步训练
python legged_lab/scripts/train.py --task=g1_21dof_amp_run --headless
```

### 策略评估

```bash
python legged_lab/scripts/play.py --task=g1_21dof_amp_walk
```

### Sim2Sim 部署（MuJoCo）

```bash
# 游戏手柄
python sim2sim/sim2sim_g1_21dof.py --policy policy/policy.onnx

# 键盘控制 (WASD+JL)
python sim2sim/sim2sim_g1_21dof_key.py --policy policy/policy.onnx
```

键位：W/S=前进/后退, A/D=横移, J/L=转向, R=归零, Q=退出

## 动捕数据处理

### PKL → CSV 转换

将 GMR 格式的 PKL 动捕数据转换为 CSV，利用 Isaac Sim 计算前向运动学（FK）生成关键身体点位置：

```bash
python legged_lab/tools/batch_pkl_to_csv_g1_21dof.py \
  --folder datasets/g1_21dof/walk_1.5 \
  --out_folder datasets/g1_21dof/walk_1.5_csv \
  --config legged_lab/tools/config/g1_21dof.yaml
```

### CSV 回放验证

```bash
python legged_lab/tools/replay_amp_g1_21dof.py \
  --csv datasets/g1_21dof/walk_1.5_csv/walk_fb_1.5.bvh_Skeleton0.csv
```

### CSV 格式（73 维）

```
root_pos(3) + root_rot(4) + lin_vel(3) + ang_vel(3)
+ joint_pos(21) + joint_vel(21) + key_body_pos(18)
```

关节顺序为 Isaac Lab articulation 顺序（L-R 配对按关节类型排列）。

## 项目结构

```
AMP_lab/
├── legged_lab/
│   ├── assets/g1_21dof/         # G1 21-DOF 机器人模型 (URDF/MJCF/STL)
│   ├── envs/g1_21dof/           # 环境和训练配置
│   │   ├── g1_21dof_amp_env.py  # AMP 环境
│   │   ├── amp_walk_cfg.py      # 行走训练配置
│   │   └── amp_run_cfg.py       # 跑步训练配置
│   ├── mdp/rewards.py           # 奖励函数
│   ├── scripts/
│   │   ├── train.py             # 训练入口
│   │   └── play.py              # 策略评估 + ONNX 导出
│   ├── tools/
│   │   ├── batch_pkl_to_csv_g1_21dof.py  # PKL→CSV 转换
│   │   ├── replay_amp_g1_21dof.py        # CSV 回放
│   │   └── config/g1_21dof.yaml          # GMR→Lab 关节映射
│   └── datasets/g1_21dof/       # 动捕数据 (PKL/CSV)
├── rsl_rl/
│   └── rsl_rl/
│       ├── runners/g1_21dof_amp_on_policy_runner.py  # AMP 训练 Runner
│       └── utils/g1_21dof_motion_loader.py           # CSV 动捕加载器
└── sim2sim/
    ├── sim2sim_g1_21dof.py      # 手柄控制 Sim2Sim
    └── sim2sim_g1_21dof_key.py  # 键盘控制 Sim2Sim
```

## 关节顺序

Isaac Lab articulation 使用 L-R 配对按关节类型排列（非 URDF 运动链顺序）：

```
left_hip_pitch, right_hip_pitch, waist_yaw,
left_hip_roll, right_hip_roll,
left_shoulder_pitch, right_shoulder_pitch,
left_hip_yaw, right_hip_yaw,
left_shoulder_roll, right_shoulder_roll,
left_knee, right_knee,
left_shoulder_yaw, right_shoulder_yaw,
left_ankle_pitch, right_ankle_pitch,
left_elbow, right_elbow,
left_ankle_roll, right_ankle_roll
```

可用 `python legged_lab/tools/print_lab_joint_body_names.py` 打印验证。

## 观测空间（72 维/步，5 帧历史）

```
ang_vel(3,×0.2) + proj_gravity(3) + command(3) + joint_pos_rel(21) + joint_vel(21,×0.05) + action(21)
```

## GMR→Lab 关节映射

GMR PKL 的关节顺序为 URDF 运动链顺序（左腿→右腿→腰→左臂→右臂），与 Isaac Lab 的 articulation 顺序不同。转换时通过 `config/g1_21dof.yaml` 做映射。

## 许可证

BSD-3-Clause
