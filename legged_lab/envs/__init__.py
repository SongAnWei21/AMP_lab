# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# ...

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg

# g1_21dof_amp
from legged_lab.envs.g1_21dof.g1_21dof_amp_env import G1_21DOF_AMP_Env
from legged_lab.envs.g1_21dof.amp_walk_cfg import (
    G1_21DOF_AMP_WalkAgentCfg,
    G1_21DOF_AMP_WalkFlatEnvCfg,
)
from legged_lab.envs.g1_21dof.amp_run_cfg import (
    G1_21DOF_AMP_RunAgentCfg,
    G1_21DOF_AMP_RunFlatEnvCfg,
)

from legged_lab.utils.task_registry import task_registry

task_registry.register("g1_21dof_amp_walk", G1_21DOF_AMP_Env, G1_21DOF_AMP_WalkFlatEnvCfg(), G1_21DOF_AMP_WalkAgentCfg())
task_registry.register("g1_21dof_amp_run", G1_21DOF_AMP_Env, G1_21DOF_AMP_RunFlatEnvCfg(), G1_21DOF_AMP_RunAgentCfg())
