"""Configs for stanford navigation environment.

Base config for stanford navigation enviornment.
"""
import numpy as np
import src.utils as utils
import datasets.nav_env as nav_env

def nav_env_base_config():
    if False:
        while True:
            i = 10
    'Returns the base config for stanford navigation environment.\n\n  Returns:\n    Base config for stanford navigation environment.\n  '
    robot = utils.Foo(radius=15, base=10, height=140, sensor_height=120, camera_elevation_degree=-15)
    env = utils.Foo(padding=10, resolution=5, num_point_threshold=2, valid_min=-10, valid_max=200, n_samples_per_face=200)
    camera_param = utils.Foo(width=225, height=225, z_near=0.05, z_far=20.0, fov=60.0, modalities=['rgb'], img_channels=3)
    data_augment = utils.Foo(lr_flip=0, delta_angle=0.5, delta_xy=4, relight=True, relight_fast=False, structured=False)
    outputs = utils.Foo(images=True, rel_goal_loc=False, loc_on_map=True, gt_dist_to_goal=True, ego_maps=False, ego_goal_imgs=False, egomotion=False, visit_count=False, analytical_counts=False, node_ids=True, readout_maps=False)
    class_map_names = ['chair', 'door', 'table']
    semantic_task = utils.Foo(class_map_names=class_map_names, pix_distance=16, sampling='uniform')
    task_params = utils.Foo(max_dist=32, step_size=8, num_steps=40, num_actions=4, batch_size=4, building_seed=0, num_goals=1, img_height=None, img_width=None, img_channels=None, modalities=None, outputs=outputs, map_scales=[1.0], map_crop_sizes=[64], rel_goal_loc_dim=4, base_class='Building', task='map+plan', n_ori=4, type='room_to_room_many', data_augment=data_augment, room_regex='^((?!hallway).)*$', toy_problem=False, map_channels=1, gt_coverage=False, input_type='maps', full_information=False, aux_delta_thetas=[], semantic_task=semantic_task, num_history_frames=0, node_ids_dim=1, perturbs_dim=4, map_resize_method='linear_noantialiasing', readout_maps_channels=1, readout_maps_scales=[], readout_maps_crop_sizes=[], n_views=1, reward_time_penalty=0.1, reward_at_goal=1.0, discount_factor=0.99, rejection_sampling_M=100, min_dist=None)
    navtask_args = utils.Foo(building_names=['area1_gates_wingA_floor1_westpart'], env_class=nav_env.VisualNavigationEnv, robot=robot, task_params=task_params, env=env, camera_param=camera_param, cache_rooms=True)
    return navtask_args