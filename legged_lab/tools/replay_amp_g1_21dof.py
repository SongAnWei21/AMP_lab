"""Replay G1 21-DOF AMP CSV motion in Isaac Sim.

CSV format (Isaac Lab articulation order, 73 dims):
    root_pos(3) + root_rot(4, xyzw) + lin_vel(3) + ang_vel(3)
    + joint_pos(21) + joint_vel(21) + key_body_pos(18)

Usage:
    python legged_lab/tools/replay_amp_g1_21dof.py --csv <path>
"""
import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay G1 21-DOF AMP CSV motion using Isaac Sim")
parser.add_argument("--csv", type=str, required=True, help="CSV file path")
parser.add_argument("--fps", type=float, default=50.0, help="Frames per second")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from legged_lab.assets.g1_21dof import G1_21DOF_CFG

# CSV columns (Isaac Lab articulation order)
#   root_pos(3) + root_rot(4, xyzw) + lin_vel(3) + ang_vel(3)
#   + joint_pos(21) + joint_vel(21) + key_body_pos(18) = 73
CSV_DIM = 73
OFF_ROOT_POS = 0
OFF_ROOT_ROT = 3
OFF_LIN_VEL = 7
OFF_ANG_VEL = 10
OFF_JOINT_POS = 13
OFF_JOINT_VEL = 34
OFF_KEY_BODY = 55

N_JOINTS = 21

KEY_LINK_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
]

EE_COLORS = [
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
]


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = G1_21DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    print(f"Loading CSV: {args_cli.csv}")
    motion_data = np.loadtxt(args_cli.csv, delimiter=",")
    num_frames = motion_data.shape[0]
    csv_dim = motion_data.shape[1]
    assert csv_dim == CSV_DIM, f"Expected {CSV_DIM} columns, got {csv_dim}"
    print(f"Frames: {num_frames}, Dims: {csv_dim}")

    dt = 1.0 / args_cli.fps

    # ---- Build simulation ----
    print("Building simulation...")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args_cli.device))
    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    num_dof = robot.num_joints
    print(f"Robot DOF: {num_dof}")

    # Print Lab joint names
    print("\nIsaac Lab articulation joint order:")
    for i, name in enumerate(robot.data.joint_names):
        print(f"  [{i:2d}] {name}")

    # Map key body names
    body_names = robot.data.body_names
    key_body_ids = []
    for name in KEY_LINK_NAMES:
        if name in body_names:
            key_body_ids.append(body_names.index(name))
        else:
            raise ValueError(f"Link '{name}' not found. Available: {body_names}")
    print(f"\nKey body indices: {key_body_ids}")

    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.8])

    # ---- Markers ----
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/KeyBodyMarkers",
        markers={
            f"ee_{i}": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=EE_COLORS[i]),
            )
            for i in range(len(KEY_LINK_NAMES))
        },
    )
    marker = VisualizationMarkers(marker_cfg)

    print("\nColor legend:")
    for i, (name, color) in enumerate(zip(KEY_LINK_NAMES, EE_COLORS)):
        print(f"  {name:30s}  RGB={color}")

    # ---- Replay loop ----
    print("\nReplaying... Close viewer window to exit.")

    frame_idx = 0
    while simulation_app.is_running():
        row = motion_data[frame_idx]

        root_pos = row[OFF_ROOT_POS : OFF_ROOT_ROT]
        root_rot_xyzw = row[OFF_ROOT_ROT : OFF_LIN_VEL]
        joint_pos = row[OFF_JOINT_POS : OFF_JOINT_VEL]    # already Lab order
        joint_vel = row[OFF_JOINT_VEL : OFF_KEY_BODY]

        # Build root state (PKL xyzw → Isaac Lab wxyz)
        root_state = torch.zeros(1, 13, device="cuda:0")
        root_state[0, 0:3] = torch.tensor(root_pos)
        root_state[0, 3] = root_rot_xyzw[3]   # w
        root_state[0, 4] = root_rot_xyzw[0]   # x
        root_state[0, 5] = root_rot_xyzw[1]   # y
        root_state[0, 6] = root_rot_xyzw[2]   # z

        # Write to robot
        joint_pos_tensor = torch.tensor(joint_pos, device="cuda:0").unsqueeze(0)
        joint_vel_tensor = torch.tensor(joint_vel, device="cuda:0").unsqueeze(0)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor)
        sim.render()
        scene.update(dt)

        # First frame debug
        if frame_idx == 0:
            print(f"\nFrame 0 - root_pos={root_pos} root_rot={root_rot_xyzw}")
            print("Joint values (Lab order):")
            joint_names = robot.data.joint_names
            for i, (name, val) in enumerate(zip(joint_names, joint_pos)):
                print(f"  [{i:2d}] {name:35s} = {val: .4f}")

        # Draw colored spheres
        body_pos_w = robot.data.body_pos_w[0]
        vis_positions = body_pos_w[key_body_ids, :].unsqueeze(0)
        marker.visualize(translations=vis_positions.reshape(-1, 3))

        frame_idx = (frame_idx + 1) % num_frames

    print("Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
