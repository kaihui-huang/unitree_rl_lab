
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def out_of_terrian(env: ManagerBasedRLEnv, threshold: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), shake_terrian_cfg: SceneEntityCfg = SceneEntityCfg("terrian")) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    robot: ContactSensor = env.scene[robot_cfg.name]
    shake_terrian: ContactSensor = env.scene[shake_terrian_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]
    terrian_pos = shake_terrian.data.root_pos_w[:, :2]
    diff = robot_pos - terrian_pos
    dis = torch.norm(diff, dim=-1)
    out_of_terrain = dis > threshold
    return out_of_terrain
