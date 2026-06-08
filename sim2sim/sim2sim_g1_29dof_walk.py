"""Sim2Sim for G1 29-DOF walk policy.

Strictly matches amp_walk_cfg.py + g1_29dof_amp_env.py + G1_29DOF_CFG.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime

from gamepad_controller import GamepadController


# ---- Exact values from amp_walk_cfg.py + G1_29DOF_CFG ----
DT = 0.005
DECIMATION = 4
ACTION_SCALE = 0.25
NUM_ACTIONS = 29
OBS_PER_STEP = 96   # 3+3+3+29+29+29
OBS_HISTORY = 5
CLIP_OBS = 100.0
CLIP_ACT = 100.0

# Observation scales (amp_walk_cfg.py normalization.obs_scales)
SCALE_ANG_VEL = 0.2
SCALE_JOINT_VEL = 0.05

# Kp / Kd — Isaac Lab articulation order (G1_29DOF_CFG actuators)
#  0: L_hip_pitch, 1: R_hip_pitch, 2: waist_yaw, 3: L_hip_roll, 4: R_hip_roll,
#  5: waist_roll, 6: L_hip_yaw, 7: R_hip_yaw, 8: waist_pitch, 9: L_knee, 10: R_knee,
# 11: L_shoulder_pitch, 12: R_shoulder_pitch, 13: L_ankle_pitch, 14: R_ankle_pitch,
# 15: L_shoulder_roll, 16: R_shoulder_roll, 17: L_ankle_roll, 18: R_ankle_roll,
# 19: L_shoulder_yaw, 20: R_shoulder_yaw, 21: L_elbow, 22: R_elbow,
# 23: L_wrist_roll, 24: R_wrist_roll, 25: L_wrist_pitch, 26: R_wrist_pitch,
# 27: L_wrist_yaw, 28: R_wrist_yaw
KP = np.array([
    100, 100,   # hip_pitch L,R
    100,        # waist_yaw
    100, 100,   # hip_roll L,R
     50,        # waist_roll
     50,  50,   # hip_yaw L,R
     50,        # waist_pitch
    100, 100,   # knee L,R
     30,  30,   # shoulder_pitch L,R
     20,  20,   # ankle_pitch L,R
     30,  30,   # shoulder_roll L,R
     20,  20,   # ankle_roll L,R
     30,  30,   # shoulder_yaw L,R
     30,  30,   # elbow L,R
     30,  30,   # wrist_roll L,R
     10,  10,   # wrist_pitch L,R
     10,  10,   # wrist_yaw L,R
], dtype=np.float32)

KD = np.array([
    4.0, 4.0,   # hip_pitch
    4.0,        # waist_yaw
    4.0, 4.0,   # hip_roll
    2.0,        # waist_roll
    2.5, 2.5,   # hip_yaw
    2.0,        # waist_pitch
    4.0, 4.0,   # knee
    2.0, 2.0,   # shoulder_pitch
    2.0, 2.0,   # ankle_pitch
    2.0, 2.0,   # shoulder_roll
    2.0, 2.0,   # ankle_roll
    2.0, 2.0,   # shoulder_yaw
    2.0, 2.0,   # elbow
    2.0, 2.0,   # wrist_roll
    1.0, 1.0,   # wrist_pitch
    1.0, 1.0,   # wrist_yaw
], dtype=np.float32)

# Default joint pos — Isaac Lab articulation order (G1_29DOF_CFG init_state)
DEFAULT_DOF_POS = np.array([
    -0.15, -0.15,   # hip_pitch L,R
     0.0,           # waist_yaw
     0.0,  0.0,     # hip_roll L,R
     0.0,           # waist_roll
     0.0,  0.0,     # hip_yaw L,R
     0.0,           # waist_pitch
     0.3,  0.3,     # knee L,R
     0.0,  0.0,     # shoulder_pitch L,R
    -0.15, -0.15,   # ankle_pitch L,R
     0.25,-0.25,    # shoulder_roll L,R
     0.0,  0.0,     # ankle_roll L,R
     0.0,  0.0,     # shoulder_yaw L,R
     1.0,  1.0,     # elbow L,R
     0.0,  0.0,     # wrist_roll L,R
     0.0,  0.0,     # wrist_pitch L,R
     0.0,  0.0,     # wrist_yaw L,R
], dtype=np.float32)

# Isaac Lab articulation order (L-R paired by joint type)
JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


class MujocoRunner:
    def __init__(self, policy_path: str, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        self.session = onnxruntime.InferenceSession(policy_path)
        self.input_name = self.session.get_inputs()[0].name

        self.step_dt = DECIMATION * DT
        self.dof_pos = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.dof_vel = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.command_vel = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.obs_buf = np.zeros(OBS_PER_STEP * OBS_HISTORY, dtype=np.float32)

        self._init_joint_mapping()
        self.gamepad = GamepadController(deadzone=0.15)

    def _init_joint_mapping(self):
        self.qpos_idx = []
        self.qvel_idx = []
        self.ctrl_idx = []

        joint_to_act = {}
        for act_id in range(self.model.nu):
            if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                joint_to_act[self.model.actuator_trnid[act_id, 0]] = act_id

        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in XML!")
            self.qpos_idx.append(self.model.jnt_qposadr[jid])
            self.qvel_idx.append(self.model.jnt_dofadr[jid])
            if jid not in joint_to_act:
                raise ValueError(f"Joint '{name}' has no actuator!")
            self.ctrl_idx.append(joint_to_act[jid])

        print(f"[INFO] {len(JOINT_NAMES)} DOF joint-actuator mapping OK.")

    def get_obs(self) -> np.ndarray:
        self.dof_pos = np.array([self.data.qpos[i] for i in self.qpos_idx])
        self.dof_vel = np.array([self.data.qvel[i] for i in self.qvel_idx])

        q = self.data.sensor("orientation").data  # [w,x,y,z]
        w, xyz = q[0], q[1:4]
        gravity = np.array([0, 0, -1]) * (2*w*w - 1) - np.cross(xyz, [0, 0, -1]) * w * 2 + xyz * np.dot(xyz, [0, 0, -1]) * 2

        cur = np.concatenate([
            self.data.sensor("angular-velocity").data * SCALE_ANG_VEL,  # 3
            gravity,                                                      # 3
            self.command_vel,                                             # 3
            self.dof_pos - DEFAULT_DOF_POS,                              # 29
            self.dof_vel * SCALE_JOINT_VEL,                              # 29
            self.action,                                                 # 29
        ]).astype(np.float32)
        cur = np.clip(cur, -CLIP_OBS, CLIP_OBS)

        self.obs_buf = np.roll(self.obs_buf, shift=-OBS_PER_STEP)
        self.obs_buf[-OBS_PER_STEP:] = cur
        return self.obs_buf

    def run(self):
        print("[INFO] Sim2Sim G1 29DOF — walk policy")
        print("[INFO] Gamepad: L stick = XY vel, R stick (L/R) = yaw. LT+B = exit.\n")

        # Warm-up
        print("[INFO] Warm-up...")
        for _ in range(OBS_HISTORY + 2):
            for _ in range(DECIMATION):
                cur_q = np.array([self.data.qpos[i] for i in self.qpos_idx])
                cur_dq = np.array([self.data.qvel[i] for i in self.qvel_idx])
                tau = KP * (DEFAULT_DOF_POS - cur_q) - KD * cur_dq
                for i, aid in enumerate(self.ctrl_idx):
                    self.data.ctrl[aid] = tau[i]
                mujoco.mj_step(self.model, self.data)
            self.action[:] = 0.0
            self.get_obs()
        print("[INFO] Ready.\n")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 45.0
            viewer.cam.elevation = -20.0

            while viewer.is_running():
                t0 = time.time()

                pad_x, pad_y, pad_yaw = self.gamepad.get_commands()
                self.command_vel = np.array([pad_x, pad_y, pad_yaw * 1.57], dtype=np.float32)
                print(f"\r[vx={self.command_vel[0]:+.2f} vy={self.command_vel[1]:+.2f} vyaw={self.command_vel[2]:+.2f}]  ", end="", flush=True)

                if self.gamepad.get_button_b() and self.gamepad.get_button_lt():
                    print("\n[INFO] LT+B — exit.")
                    break

                obs = self.get_obs()
                raw = self.session.run(None, {self.input_name: obs.reshape(1, -1)})[0].flatten()[:29]
                self.action[:] = np.clip(raw, -CLIP_ACT, CLIP_ACT)
                target = self.action * ACTION_SCALE + DEFAULT_DOF_POS

                for _ in range(DECIMATION):
                    cur_q = np.array([self.data.qpos[i] for i in self.qpos_idx])
                    cur_dq = np.array([self.data.qvel[i] for i in self.qvel_idx])
                    tau = KP * (target - cur_q) - KD * cur_dq
                    for i, aid in enumerate(self.ctrl_idx):
                        self.data.ctrl[aid] = tau[i]
                    mujoco.mj_step(self.model, self.data)

                viewer.cam.lookat[:] = self.data.qpos[0:3]
                viewer.sync()

                delay = self.step_dt - (time.time() - t0)
                if delay > 0:
                    time.sleep(delay)

        print("[INFO] Exit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="ONNX policy path")
    parser.add_argument("--model", type=str,
                        default="/home/saw/AMP/AMP_lab/legged_lab/assets/g1_29dof/g1_29dof.xml",
                        help="MuJoCo XML path")
    args = parser.parse_args()
    runner = MujocoRunner(args.policy, args.model)
    runner.run()
