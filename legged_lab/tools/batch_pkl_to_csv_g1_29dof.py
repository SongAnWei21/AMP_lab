import argparse
import sys
import os
import glob
import pickle

import numpy as np
import torch
import yaml

# # Fix pickle compatibility: pickle created with NumPy 2.x references numpy._core
# if "numpy._core" not in sys.modules and hasattr(np, "core"):
#     sys.modules["numpy._core"] = np.core

import torch

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from legged_lab.assets.g1_29dof import G1_29DOF_CFG

KEY_LINKS = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


@configclass
class FkSceneCfg(InteractiveSceneCfg):
    """Minimal scene for FK computation (headless, no rendering)."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q, r):
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = r
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_rotate_inverse(q, v):
    q_conj = quat_conjugate(q)
    v_quat = np.array([v[0], v[1], v[2], 0.0])
    return quat_multiply(quat_multiply(q_conj, v_quat), q)[:3]


def compute_lin_vel_body(root_pos, root_rot, fps):
    num_frames = root_pos.shape[0]
    lin_vel_world = np.zeros((num_frames, 3))
    if num_frames > 1:
        lin_vel_world[1:] = (root_pos[1:] - root_pos[:-1]) * fps
        lin_vel_world[0] = lin_vel_world[1]
    lin_vel_body = np.zeros((num_frames, 3))
    for t in range(num_frames):
        lin_vel_body[t] = quat_rotate_inverse(root_rot[t, [1, 2, 3, 0]], lin_vel_world[t])
    return lin_vel_body


def compute_ang_vel_body(root_rot, fps):
    num_frames = root_rot.shape[0]
    ang_vel_body = np.zeros((num_frames, 3))
    for t in range(num_frames - 1):
        q0 = root_rot[t]
        q1 = root_rot[t + 1]
        dq = quat_multiply(quat_conjugate(q0), q1)
        w = np.clip(dq[3], -1.0, 1.0)
        angle = 2.0 * np.arccos(np.abs(w))
        sin_half = np.sin(angle / 2.0)
        if abs(sin_half) > 1e-8:
            axis = dq[:3] / sin_half
        else:
            axis = np.zeros(3)
        if w < 0:
            axis = -axis
        ang_vel_body[t] = axis * angle * fps
    ang_vel_body[-1] = ang_vel_body[-2]
    return ang_vel_body


def main():
    parser = argparse.ArgumentParser(
        description="G1 29-DOF PKL -> AMP CSV (lin_vel, ang_vel, joint_pos, joint_vel, key_body_pos)"
    )
    parser.add_argument("--folder", type=str, required=True, help="PKL folder path")
    parser.add_argument("--out_folder", type=str, default=None, help="Output CSV folder (default: <folder>_csv)")
    parser.add_argument("--config", type=str, default="legged_lab/tools/config/g1_29dof.yaml", help="YAML config for GMR→Lab joint mapping")
    args = parser.parse_args()

    pkl_folder = args.folder
    out_folder = args.out_folder or (pkl_folder.rstrip("/") + "_csv")
    os.makedirs(out_folder, exist_ok=True)

    pkl_files = glob.glob(os.path.join(pkl_folder, "*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found under {pkl_folder}.")
        return

    # Load GMR→Lab mapping
    with open(args.config) as f_cfg:
        mapping_cfg = yaml.safe_load(f_cfg)
    gmr_names = mapping_cfg["gmr_dof_names"]
    lab_names = mapping_cfg["lab_dof_names"]
    gmr_to_lab = [gmr_names.index(name) for name in lab_names]
    print(f"Loaded GMR→Lab mapping ({len(gmr_to_lab)} joints)")

    # ---- Build simulation ----
    print("Starting Isaac Sim for forward kinematics...")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 50.0, device="cuda:0"))
    scene_cfg = FkSceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]

    # Map key body names to indices
    body_names = robot.data.body_names
    key_body_ids = []
    for name in KEY_LINKS:
        if name in body_names:
            key_body_ids.append(body_names.index(name))
        else:
            print(f"Link '{name}' not found in URDF!")
            return
    print(f"Mapped {len(KEY_LINKS)} key body links: {KEY_LINKS}")
    n_key_bodies = len(KEY_LINKS)
    num_dof = robot.num_joints
    print(f"Robot DOF: {num_dof}")

    for pkl_file in pkl_files:
        print(f"\nProcessing: {os.path.basename(pkl_file)}")
        with open(pkl_file, "rb") as f:
            motion_data = pickle.load(f)

        root_pos_data = motion_data["root_pos"]
        root_rot_data = motion_data["root_rot"]
        dof_pos_data = motion_data["dof_pos"]
        fps = motion_data.get("fps", 50.0)
        dt = 1.0 / fps
        num_frames = root_pos_data.shape[0]

        # --- lin_vel (use PKL pre-computed body-frame value if available) ---
        if "root_vel_body" in motion_data:
            lin_vel_data = motion_data["root_vel_body"]
        else:
            lin_vel_data = compute_lin_vel_body(root_pos_data, root_rot_data, fps)

        # --- ang_vel (use PKL pre-computed local value if available) ---
        if "root_rot_vel" in motion_data:
            ang_vel_data = motion_data["root_rot_vel"]
        else:
            ang_vel_data = compute_ang_vel_body(root_rot_data, fps)

        # --- joint_vel (use PKL pre-computed if available) ---
        if "dof_vel" in motion_data:
            dof_vel_data = motion_data["dof_vel"]
        else:
            dof_vel_data = np.zeros_like(dof_pos_data)
            if num_frames > 1:
                dof_vel_data[1:] = (dof_pos_data[1:] - dof_pos_data[:-1]) * fps
                dof_vel_data[0] = dof_vel_data[1]

        # Remap joint_pos and joint_vel from GMR to Lab order
        dof_pos_data = dof_pos_data[:, gmr_to_lab]
        dof_vel_data = dof_vel_data[:, gmr_to_lab]

        # --- key_body_pos (FK via Isaac Sim) ---
        root_state = torch.zeros(1, 13, device="cuda:0")
        root_state[:, 3] = 1.0  # identity quaternion
        joint_vel_tensor = torch.zeros(1, num_dof, device="cuda:0")

        key_body_pos_data = np.zeros((num_frames, n_key_bodies * 3), dtype=np.float32)
        for t in range(num_frames):
            dof_pos_lab = dof_pos_data[t]
            joint_pos_tensor = torch.tensor(dof_pos_lab, device="cuda:0").unsqueeze(0)

            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor)
            sim.render()
            scene.update(dt)

            body_pos_w = robot.data.body_pos_w[0]  # (num_bodies, 3)
            for i, b_id in enumerate(key_body_ids):
                key_body_pos_data[t, i * 3 : (i + 1) * 3] = body_pos_w[b_id, :].cpu().numpy()

            if (t + 1) % 500 == 0:
                print(f"  FK progress: {t + 1}/{num_frames}")

        # Output: root_pos(3) + root_rot(4, xyzw) + lin_vel(3) + ang_vel(3)
        #         + joint_pos(29) + joint_vel(29) + key_body_pos(18) = 89
        amp_motion_data = np.hstack((
            root_pos_data,      # 3
            root_rot_data,      # 4  (xyzw)
            lin_vel_data,       # 3
            ang_vel_data,       # 3
            dof_pos_data,       # 29
            dof_vel_data,       # 29
            key_body_pos_data,  # 18
        ))
        total_dim = 3 + 4 + 3 + 3 + dof_pos_data.shape[1] + dof_vel_data.shape[1] + n_key_bodies * 3
        print(f"  CSV dims: {amp_motion_data.shape[1]} (expected: {total_dim})")

        base_name = os.path.splitext(os.path.basename(pkl_file))[0]
        output_csv = os.path.join(out_folder, f"{base_name}.csv")
        np.savetxt(output_csv, amp_motion_data, delimiter=",", fmt="%.6f")
        print(f"  Saved: {output_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
    simulation_app.close()
