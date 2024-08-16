# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG 
from omni.isaac.lab_assets import IRIS_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

# obstacle
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim.spawners.shapes import spawn_cylinder, CylinderCfg

# sensor
from omni.isaac.lab.sensors.ray_caster import RayCasterCfg, RayCaster, patterns

# shortest path
from omni.isaac.debug_draw import _debug_draw
from sklearn.neighbors import NearestNeighbors
import heapq
from scipy.interpolate import splprep, splev
import numpy as np


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0 #10.0
    decimation = 2 
    num_actions = 4
    num_observations = 12
    num_states = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)
    
    # sensor - ray caster attached to the base of robot 1 
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=False,
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        # pattern_cfg=patterns.LidarPatternCfg(channels=2, vertical_fov_range=[10.0, 20.0], horizontal_fov_range=[0.0, 180.0],horizontal_res=1.0),     
        pattern_cfg=patterns.BpearlPatternCfg(vertical_ray_angles=[2.3125]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    

    # robot
    robot: ArticulationCfg = IRIS_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 6.0
    moment_scale = 1.0 #0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.5 #-0.01
    distance_to_goal_reward_scale = 5.0 #15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        
        #Obstacle position
        self.obstacle_1_pos = torch.tensor([3.5, 1.5, 1.5], device=self.device)
        self.obstacle_2_pos = torch.tensor([2.5, -1.5, 1.5], device=self.device)
        self.future_traj_steps = 4

        self.shortest_path = GuildingPath(progress_buf=self.episode_length_buf, 
                                          env_origins=self._terrain.env_origins, 
                                          obstacle_1_pos=self.obstacle_1_pos, 
                                          obstacle_2_pos=self.obstacle_2_pos, 
                                          number_env=self.num_envs, 
                                          device=self.device)
        self.shortest_path.generateGuidingPath()
        self.shortest_path.plot_shortest_path() 

        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _add_obstacles(self):
        # Rigid Object
        # cylinder_cfg = sim_utils.CylinderCfg(
        #     radius=0.5,
        #     height=3.0,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        #     mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
        #     collision_props=sim_utils.CollisionPropertiesCfg(),
        #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        # )
        # # cylinder_cfg.func("/World/envs/env_.*/Cylinder_1", cylinder_cfg, translation=obstacle_1_pos)
    
        # cylinder = RigidObjectCfg(
        #     prim_path="/World/envs/env_.*/Cylinder_1",
        #     spawn=cylinder_cfg,
        #     init_state=RigidObjectCfg.InitialStateCfg(),
        # )
        # cylinder_object = RigidObject(cfg=cylinder)

        cylinder_cfg = CylinderCfg(
            radius=0.5,
            height=3.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2)
        )
        cylinder_1 = spawn_cylinder(prim_path="/World/envs/env_.*/Cylinder_1", cfg=cylinder_cfg, translation=(3.5, 1.5, 1.5))
        return cylinder_1


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.obstacle_1 = self._add_obstacles()

        # self.scene.rigid_objects["obstacles"] = self.obstacle_1

        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        
        # self.scene.rigid_objects["obstacles"] = self._terrain
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # self.obstacle_1.write_root_pose_to_sim(torch.tensor([3.5, 1.5, 1.5]))
        


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        # (self, steps: int, env_ids=None, step_size=1)
       
        self.shortest_path.progress_buf = self.episode_length_buf
        self.target_pos = self.shortest_path.compute_shortest_traj(steps = self.future_traj_steps, step_size=5)
        # print("\n")
        # print(self.target_pos[:,0,:].dtype)
        # print(self._robot.data.root_state_w[:, 3:7].dtype)
        # print("\n")
        
        desired_pos_b, _ = subtract_frame_transforms(
            # self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self.target_pos[:,0,:].to(torch.float32)
        )
        # print("-------------------------------")
        # print(self.scene["height_scanner"])
        # print("Received max height value: ", torch.min(self.scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        # print("-------------------------------")
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        # print(self.target_pos[:,0,:])
        # print("\n")
        # print(self._robot.data.root_pos_w-self._terrain.env_origins)
        # self._robot_pos = self._robot.data.root_pos_w - self._terrain.env_origins
        # self.distance_to_goal = torch.linalg.norm(self.target_pos[:,0,:] - self._robot_pos , dim=1)
        # print(self.distance_to_goal)
        distance_to_goal_mapped = 1 - torch.tanh(self.distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # print("progress ",self.episode_length_buf)
        ones = torch.ones_like(self.reset_buf)
        died = torch.zeros_like(self.reset_buf)
    
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        
        self._robot_pos = self._robot.data.root_pos_w - self._terrain.env_origins
        self.distance_to_goal = torch.linalg.norm(self.target_pos[:,0,:] - self._robot_pos, dim=1)
        died = torch.where(self.distance_to_goal > 0.3, ones, died)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        # Sample new commands
        # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        # self.goal_pos_visualizer.visualize(self._desired_pos_w)
        self.goal_pos_visualizer.visualize(self.target_pos[:,0,:] - self._terrain.env_origins)



class GuildingPath:
    def __init__(self, progress_buf, env_origins ,number_env ,device, obstacle_1_pos = (3.0,2.0,1.5), obstacle_2_pos = (3.0,2.0,1.5),future_timestep = 4):
        
        self.progress_buf = progress_buf

        
        self.future_traj_steps = future_timestep
        self.env_origin_pos = env_origins
        self.obstacle_1_pos = obstacle_1_pos
        self.obstacle_2_pos = obstacle_2_pos
        
        self.num_envs = number_env
        self.device = device

        
        self.cylinder_center = torch.tensor([0.0 , 1.0],device=self.device)
        self.cylinder_radius = torch.tensor([1.0],device=self.device) # meter
        
        # print("debug env pos shape",self.env_origin_pos.shape)           # (4015,3)
        self.central_env_idx = self.env_origin_pos.norm(dim=-1).argmin()   # env 2015
        # print("central idx  ",self.central_env_idx)

        self.origin = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.t = torch.zeros(self.num_envs, self.future_traj_steps, device=self.device)
        
        self.num_samples = 4000 
        self.area_size = 10  #meter
        
        
        self.k = 5
        self.cylinder_radius = 1 # meter
        
        self.start = torch.tensor([0.0, 0.0],device=self.device)   # it's should be the same robot spawn 
        self.goal = torch.tensor([4.0, 4.0],device=self.device)


        self.render = True

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.draw.clear_lines()
        print("done")


    # Check if a point is inside the cylinder obstacle
    def _in_cylinder(self, p, center, radius):
        return torch.norm(p.to(self.device) - center.to(self.device), dim=-1) < radius
    
    # B-spline interpolation
    def _bspline_path(self,path, degree=3, num_points=4000):
        tck, u = splprep([path[:, 0].numpy(), path[:, 1].numpy()], s=0, k=degree)
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_new, y_new = splev(u_new, tck)
        return x_new, y_new
    
    def _dijkstra(self, samples, edges, start_idx, goal_idx):
        graph = {i: [] for i in range(len(samples))}
        for edge in edges:
            graph[edge[0]].append((torch.norm(samples[edge[0]] - samples[edge[1]]).item(), edge[1]))
            graph[edge[1]].append((torch.norm(samples[edge[0]] - samples[edge[1]]).item(), edge[0]))
        
        queue = [(0, start_idx)]
        distances = {i: float('inf') for i in range(len(samples))}
        distances[start_idx] = 0
        prev = {i: None for i in range(len(samples))}
        
        while queue:
            curr_dist, curr_node = heapq.heappop(queue)
            
            if curr_dist > distances[curr_node]:
                continue
            
            for weight, neighbor in graph[curr_node]:
                distance = curr_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    prev[neighbor] = curr_node
                    heapq.heappush(queue, (distance, neighbor))
        
        # Reconstruct the path
        path = []
        curr = goal_idx
        while prev[curr] is not None:
            path.append(curr)
            curr = prev[curr]
        path.append(start_idx)
        return path[::-1]
    
    def generateGuidingPath(self):
    
        obs0_position  = self.obstacle_1_pos  #still bug | dim is (1)
        obs1_position  = self.obstacle_2_pos
        # print("Position is:", obs0_position - self._env_pos)
        # print("Shape Position is:", obs0_position[0][:-1].shape)
        # obs0_position = obs0_position - self.env_origin_pos
        # obs1_position = obs1_position - self.env_origin_pos
        # print("Orientation is:", oritentation)
        

        random_samples = torch.rand(self.num_samples, 2, device=self.device)
        samples = (self.area_size * random_samples) - (self.area_size/2)
        # print("cylinder center = ",cylinder_center)
        # print("cylinder center = ",cylinder_center.shape)

        # print("samples center = ",samples.shape)
        samples = samples[~self._in_cylinder(samples, self.cylinder_center, self.cylinder_radius)]
        samples = samples[~self._in_cylinder(samples, obs0_position[:-1].to(self.device), self.cylinder_radius)]
        samples = samples[~self._in_cylinder(samples, obs1_position[:-1].to(self.device), self.cylinder_radius)]
        
        samples = torch.cat([self.start.unsqueeze(0), self.goal.unsqueeze(0), samples], dim=0)
        
        # Create edges based on k-nearest neighbors
        samples = samples.cpu()
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(samples)
        distances, indices = neighbors.kneighbors(samples)

        # Create graph
        edges = []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:
                    p1, p2 = samples[i], samples[j]
                    if (not self._in_cylinder((p1 + p2) / 2, self.cylinder_center, self.cylinder_radius)):
                        edges.append((i, j))
                        

        # Get the shortest path
        path_indices = self._dijkstra(samples , edges, 0, 1)
        path = samples[path_indices]

        spline_x, spline_y = self._bspline_path(path)
        
        spline_x = torch.tensor(spline_x, device=self.device)
        spline_y = torch.tensor(spline_y, device=self.device)
        spline_z = torch.ones_like(spline_x, device=self.device)

        # Combine x, y, z into a single tensor
        self.spline_xyz = torch.stack((spline_x, spline_y, spline_z), dim=1)  #size [300,3]
        self.duplicated_spline_xyz = self.spline_xyz.unsqueeze(0).repeat(self.num_envs, 1, 1)   #[num_env,length_bspline,xyz]
        
        self.spline_xyz = self.spline_xyz + self.env_origin_pos[self.central_env_idx]
        # print("spline_xyz shape",self.spline_xyz.shape)
        point_list_0 = self.spline_xyz[:-1].tolist()   # cut the endding point to make a line ex. whose line is 1,2,3,4; point_list_0 = 1,2,3  
        point_list_1 = self.spline_xyz[1:].tolist()    # cut the starting point to make a line ex. whose line is 1,2,3,4; point_list_1 = 2,3,4   then the line is 1-2, 2-3, 3-4
        # print(len(point_list_0))
        debug_plot = False
        if debug_plot == True:
            colors = [(1.0, 1.0, 0.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [2 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes) #draw the line



    def compute_shortest_traj(self, steps: int, env_ids=None, step_size=1):
        if env_ids is None:
            env_ids = ...

        # print("progress ",self.progress_buf)
        self.t = self.progress_buf[env_ids].unsqueeze(-1).long()*4 + step_size * torch.arange(steps, device=self.device, dtype=torch.long)
        self.traj_target_spline = self.duplicated_spline_xyz[torch.arange(self.duplicated_spline_xyz.size(0)).unsqueeze(1), self.t]
        
        return self.origin + self.traj_target_spline
    
    def plot_shortest_path(self):
        traj_vis =  self.duplicated_spline_xyz[:,:,:] + self.origin    # !!!! must edit to max length of traj_target_spline
        # traj_vis = self._compute_shortest_traj(steps = 1000, env_ids = self.central_env_idx)
        traj_vis = traj_vis + self.env_origin_pos[self.central_env_idx]
        # self._terrain.env_origins


        plot_point_list_0 = traj_vis[self.central_env_idx ,:-1 ,:]
        plot_point_list_1 = traj_vis[self.central_env_idx ,1:  ,:]
        # print(plot_point_list_0.shape)

        plot_point_list_0 = plot_point_list_0.tolist()
        plot_point_list_1 = plot_point_list_1.tolist()

        colors = [(1.0, 0.0, 0.0, 1.0) for _ in range(len(plot_point_list_0))]
        sizes = [2 for _ in range(len(plot_point_list_1))]
        self.draw.draw_lines(plot_point_list_0, plot_point_list_1, colors, sizes) #draw the line
    