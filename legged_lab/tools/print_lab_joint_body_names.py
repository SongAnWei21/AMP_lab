"""Print Isaac Lab joint/body names and order for G1 21-DOF robot.

Usage:
    python legged_lab/tools/print_lab_joint_body_names.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print G1 21-DOF joint and body names from Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from legged_lab.assets.g1_21dof import G1_21DOF_CFG


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))

    prim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])
    robot = Articulation(G1_21DOF_CFG.replace(prim_path="/World/Origin1/Robot"))
    sim.reset()

    # --- Joint names (Articulation order) ---
    joint_names = robot.joint_names
    print(f"\nG1 21-DOF articulation joint names ({len(joint_names)} DOFs):")
    for i, name in enumerate(joint_names):
        print(f"  [{i:2d}] {name}")

    # --- Body names and masses ---
    body_names = robot.body_names
    body_masses = robot.data.default_mass[0]
    print(f"\nG1 21-DOF articulation body names ({len(body_names)} bodies):")
    for i, (name, mass) in enumerate(zip(body_names, body_masses)):
        print(f"  [{i:2d}] {name:40s}  mass={mass:.4f}")

    simulation_app.close()


if __name__ == "__main__":
    main()
