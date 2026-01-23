# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""
import torch
import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import HfDiscreteObstaclesTerrainCfg


# OBSTACLE_RAND_POS = torch.randint(-10, 10, (10, 3), dtype=torch.float32)
random_obs = True
if random_obs:
    num_obs = 30
    # obs_pos_x = torch.randint(-10, 10, (num_obs, 1), dtype=torch.float32)
    # obs_pos_y = torch.randint(0, 15, (num_obs, 1), dtype=torch.float32)

    obs_pos_x = torch.rand(num_obs, 1) * 15 - 7.5
    obs_pos_y = torch.rand(num_obs, 1) * 15 - 7.5
    obs_pos_z = torch.ones_like(obs_pos_y, dtype=torch.float32)

    mask = ~((obs_pos_x[:, 0].abs() <= 1.5) & (obs_pos_y[:, 0].abs() <= 1.5))
    obs_pos_x = obs_pos_x[mask]
    obs_pos_y = obs_pos_y[mask]
    obs_pos_z = obs_pos_z[mask]
    
    OBSTACLE_RAND_POS = torch.cat((obs_pos_x, obs_pos_y, obs_pos_z), 1)

    obs_pos = [tuple(row.tolist()) for row in OBSTACLE_RAND_POS]

else:
    obs_pos = [(-0.5, 4.5, 1), (0.5, 6.5, 1), (-3, 8, 1), (1.7, 10, 1), (-1.2, 12, 1), (1.5, 12, 1)]
    OBSTACLE_RAND_POS = obs_pos

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(40.0, 40.0),
    border_width=0.1,
    border_height=5.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),

        # "obstacles": HfDiscreteObstaclesTerrainCfg(
        #     horizontal_scale=0.1,
        #     vertical_scale=0.1,
        #     border_width=0.0,
        #     num_obstacles=4096,
        #     obstacle_height_mode="choice",
        #     obstacle_width_range=(0.4, 1.1),
        #     obstacle_height_range=(1.3, 4.5),
        #     # obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
        #     platform_width=0.0,
        # ),
                    
        "pole" : terrain_gen.customFlatAndPoleTerrainCfg(
            proportion=0.1, pole_pos=obs_pos,
        ),
    },
)


"""Rough terrains configuration."""
