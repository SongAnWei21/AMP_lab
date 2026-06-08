"""MuJoCo replay for G1 21-DOF PKL motion data.

Plays back raw PKL files directly in MuJoCo viewer.

Usage:
    python legged_lab/tools/mujoco_replay_pkl_g1_21dof.py --pkl <path.pkl>
    python legged_lab/tools/mujoco_replay_pkl_g1_21dof.py --pkl <path.pkl> --fps 30
"""
import argparse
import io
import pickle
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Fix pickle compatibility with NumPy 2.x pickles loaded in different NumPy version
try:
    import numpy._core as _nc
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _nc
    if "numpy._core.multiarray" not in sys.modules:
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
except ImportError:
    pass


class NumpyCompatUnpickler(pickle.Unpickler):
    """Handle NumPy 2.x namespace changes during unpickling."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pkl(path: str) -> dict:
    """Load a pickle file with NumPy 2.x compatibility."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return pickle.loads(raw)
    except (AttributeError, ModuleNotFoundError, SystemError, TypeError):
        # Fallback: use custom unpickler
        return NumpyCompatUnpickler(io.BytesIO(raw)).load()

# GMR joint order (kinematic chain, 21 DOFs) — matches MuJoCo XML joint order
GMR_JOINT_NAMES = [
    # Left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint",
    # Left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    # Right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


def main():
    parser = argparse.ArgumentParser(description="MuJoCo PKL replay for G1 21-DOF")
    parser.add_argument("--pkl", type=str, required=True, help="PKL file path")
    parser.add_argument("--model", type=str,
                        default="legged_lab/assets/g1_21dof/g1_21dof.xml",
                        help="MuJoCo XML model path")
    parser.add_argument("--fps", type=float, default=None, help="Replay FPS (default: from PKL or 50)")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    # Load PKL
    print(f"Loading PKL: {args.pkl}")
    data = load_pkl(args.pkl)

    root_pos = data["root_pos"]       # (N, 3)
    root_rot = data["root_rot"]       # (N, 4) xyzw
    dof_pos = data["dof_pos"]         # (N, 21) GMR order
    pkl_fps = data.get("fps", 50.0)

    num_frames = root_pos.shape[0]
    print(f"Frames: {num_frames}, PKL FPS: {pkl_fps}")

    replay_fps = args.fps or pkl_fps
    dt = 1.0 / replay_fps

    # Load MuJoCo model
    print(f"Loading model: {args.model}")
    model = mujoco.MjModel.from_xml_path(args.model)
    data_mj = mujoco.MjData(model)

    # Map GMR joint names → MuJoCo qpos/qvel indices
    qpos_ids = []
    qvel_ids = []
    for name in GMR_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            print(f"Warning: Joint '{name}' not found in model, skipping.")
            continue
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])

    print(f"Mapped {len(qpos_ids)} / 21 joints.")

    # Print joint list for verification
    print("\nJoint mapping (GMR → MuJoCo):")
    for i, name in enumerate(GMR_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1:
            print(f"  [{i:2d}] {name:35s} qpos={model.jnt_qposadr[jid]:2d}")

    print(f"\nReplaying at {replay_fps:.1f} FPS... Close viewer to exit.\n")

    frame_idx = 0
    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -15.0

        while viewer.is_running():
            t0 = time.time()

            # Set root position
            data_mj.qpos[0] = root_pos[frame_idx, 0]
            data_mj.qpos[1] = root_pos[frame_idx, 1]
            data_mj.qpos[2] = root_pos[frame_idx, 2]

            # Set root orientation (PKL xyzw → MuJoCo wxyz)
            rx, ry, rz, rw = root_rot[frame_idx]
            data_mj.qpos[3] = rw
            data_mj.qpos[4] = rx
            data_mj.qpos[5] = ry
            data_mj.qpos[6] = rz

            # Set joint positions (GMR order → MuJoCo)
            for i, (qid, vid) in enumerate(zip(qpos_ids, qvel_ids)):
                data_mj.qpos[qid] = dof_pos[frame_idx, i]

            # Forward kinematics
            mujoco.mj_forward(model, data_mj)

            print(f"\rFrame: {frame_idx + 1}/{num_frames}", end="", flush=True)

            viewer.cam.lookat[:] = data_mj.qpos[0:3]
            viewer.sync()

            # Frame timing
            delay = dt - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)

            # Advance frame
            frame_idx += 1
            if frame_idx >= num_frames:
                if args.loop:
                    frame_idx = 0
                else:
                    print("Playback done. Close viewer to exit.")
                    # Pause at end, keep viewer open
                    while viewer.is_running():
                        time.sleep(0.1)
                    break


if __name__ == "__main__":
    main()
