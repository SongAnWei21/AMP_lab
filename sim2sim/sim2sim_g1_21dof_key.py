"""Sim2Sim G1 21-DOF — keyboard control (pynput, works regardless of window focus).

W/S = forward/back ±0.1, A/D = strafe ±0.1, J/L = yaw ±0.1
R = reset, Q = quit.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime
from pynput import keyboard

# ---- Exact values from velocity_cfg.py + G1_21DOF_CFG ----
DT = 0.005
DECIMATION = 4
ACTION_SCALE = 0.25
NUM_ACTIONS = 21
OBS_PER_STEP = 72
OBS_HISTORY = 5
CLIP_OBS = 100.0
CLIP_ACT = 100.0
SCALE_ANG_VEL = 0.2
SCALE_JOINT_VEL = 0.05

KP = np.array([
    100, 100, 100, 100, 100, 30, 30, 50, 50, 30, 30,
    100, 100, 30, 30, 20, 20, 30, 30, 20, 20,
], dtype=np.float32)

KD = np.array([
    4.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.5, 2.5, 2.0, 2.0,
    4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
], dtype=np.float32)

DEFAULT_DOF_POS = np.array([
    -0.15, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, -0.25,
    0.3, 0.3, 0.0, 0.0, -0.15, -0.15, 1.0, 1.0, 0.0, 0.0,
], dtype=np.float32)

JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]


class CmdState:
    """Thread-safe command state, updated by pynput listener."""
    def __init__(self):
        self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0
        self.running = True

    def on_press(self, key):
        try:
            c = key.char
            if c == 'w':
                self.vx += 0.1
            elif c == 's':
                self.vx -= 0.1
            elif c == 'd':
                self.vy += 0.1
            elif c == 'a':
                self.vy -= 0.1
            elif c == 'j':
                self.vyaw += 0.1
            elif c == 'l':
                self.vyaw -= 0.1
            elif c == 'r':
                self.vx = self.vy = self.vyaw = 0.0
            elif c == 'q':
                self.running = False
        except AttributeError:
            pass


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
        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.obs_buf = np.zeros(OBS_PER_STEP * OBS_HISTORY, dtype=np.float32)
        self._init_joint_mapping()
        self.cmd = CmdState()
        self._listener = keyboard.Listener(on_press=self.cmd.on_press)
        self._listener.start()

    def _init_joint_mapping(self):
        self.qpos_idx, self.qvel_idx, self.ctrl_idx = [], [], []
        joint_to_act = {}
        for act_id in range(self.model.nu):
            if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                joint_to_act[self.model.actuator_trnid[act_id, 0]] = act_id
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found!")
            self.qpos_idx.append(self.model.jnt_qposadr[jid])
            self.qvel_idx.append(self.model.jnt_dofadr[jid])
            if jid not in joint_to_act:
                raise ValueError(f"Joint '{name}' has no actuator!")
            self.ctrl_idx.append(joint_to_act[jid])
        print(f"[INFO] {len(JOINT_NAMES)} DOF mapping OK.")

    def get_obs(self):
        self.dof_pos = np.array([self.data.qpos[i] for i in self.qpos_idx])
        self.dof_vel = np.array([self.data.qvel[i] for i in self.qvel_idx])
        q = self.data.sensor("orientation").data
        w, xyz = q[0], q[1:4]
        gravity = np.array([0, 0, -1]) * (2*w*w - 1) - np.cross(xyz, [0, 0, -1]) * w * 2 + xyz * np.dot(xyz, [0, 0, -1]) * 2
        cur = np.concatenate([
            self.data.sensor("angular-velocity").data * SCALE_ANG_VEL,
            gravity,
            self.command_vel,
            self.dof_pos - DEFAULT_DOF_POS,
            self.dof_vel * SCALE_JOINT_VEL,
            self.action,
        ]).astype(np.float32)
        cur = np.clip(cur, -CLIP_OBS, CLIP_OBS)
        self.obs_buf = np.roll(self.obs_buf, shift=-OBS_PER_STEP)
        self.obs_buf[-OBS_PER_STEP:] = cur
        return self.obs_buf

    def run(self):
        print("[INFO] Sim2Sim G1 21DOF — Keyboard (pynput)")
        print("[INFO] W/S=±vx  A/D=±vy  J/L=±vyaw  R=reset  Q=quit")
        print("[INFO] Focus the MuJoCo window, type in terminal!\n")

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

            while viewer.is_running() and self.cmd.running:
                t0 = time.time()

                self.command_vel = np.array([self.cmd.vx, self.cmd.vy, self.cmd.vyaw], dtype=np.float32)
                print(f"\r[vx={self.cmd.vx:+.1f} vy={self.cmd.vy:+.1f} vyaw={self.cmd.vyaw:+.1f}]  ", end="", flush=True)

                obs = self.get_obs()
                raw = self.session.run(None, {self.input_name: obs.reshape(1, -1)})[0].flatten()[:21]
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

        self._listener.stop()
        print("[INFO] Exit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="ONNX policy path")
    parser.add_argument("--model", type=str,
                        default="/home/saw/AMP/AMP_lab/legged_lab/assets/g1_21dof/g1_21dof.xml",
                        help="MuJoCo XML path")
    args = parser.parse_args()
    runner = MujocoRunner(args.policy, args.model)
    runner.run()
