import open3d as o3d
import os
import sys
import json
from os.path import isfile, join, splitext, dirname, basename
from warnings import warn
from data_loader import lounge_data_loader, bedroom_data_loader, jackjack_data_loader

def extract_rgbd_frames(rgbd_video_file):
    if False:
        i = 10
        return i + 15
    '\n    Extract color and aligned depth frames and intrinsic calibration from an\n    RGBD video file (currently only RealSense bag files supported). Folder\n    structure is:\n        <directory of rgbd_video_file/<rgbd_video_file name without extension>/\n            {depth/00000.jpg,color/00000.png,intrinsic.json}\n    '
    frames_folder = join(dirname(rgbd_video_file), basename(splitext(rgbd_video_file)[0]))
    path_intrinsic = join(frames_folder, 'intrinsic.json')
    if isfile(path_intrinsic):
        warn(f'Skipping frame extraction for {rgbd_video_file} since files are present.')
    else:
        rgbd_video = o3d.t.io.RGBDVideoReader.create(rgbd_video_file)
        rgbd_video.save_frames(frames_folder)
    with open(path_intrinsic) as intr_file:
        intr = json.load(intr_file)
    depth_scale = intr['depth_scale']
    return (frames_folder, path_intrinsic, depth_scale)

def set_default_value(config, key, value):
    if False:
        for i in range(10):
            print('nop')
    if key not in config:
        config[key] = value

def initialize_config(config):
    if False:
        while True:
            i = 10
    set_default_value(config, 'depth_map_type', 'redwood')
    set_default_value(config, 'n_frames_per_fragment', 100)
    set_default_value(config, 'n_keyframes_per_n_frame', 5)
    set_default_value(config, 'depth_min', 0.3)
    set_default_value(config, 'depth_max', 3.0)
    set_default_value(config, 'voxel_size', 0.05)
    set_default_value(config, 'depth_diff_max', 0.07)
    set_default_value(config, 'depth_scale', 1000)
    set_default_value(config, 'preference_loop_closure_odometry', 0.1)
    set_default_value(config, 'preference_loop_closure_registration', 5.0)
    set_default_value(config, 'tsdf_cubic_size', 3.0)
    set_default_value(config, 'icp_method', 'color')
    set_default_value(config, 'global_registration', 'ransac')
    set_default_value(config, 'python_multi_threading', True)
    set_default_value(config, 'max_iterations', 5)
    set_default_value(config, 'sdf_trunc', 0.04)
    set_default_value(config, 'block_count', 40000)
    set_default_value(config, 'distance_threshold', 0.07)
    set_default_value(config, 'fitness_threshold', 0.3)
    set_default_value(config, 'regularizer_weight', 1)
    set_default_value(config, 'method', 'slac')
    set_default_value(config, 'device', 'CPU:0')
    set_default_value(config, 'save_output_as', 'pointcloud')
    set_default_value(config, 'folder_slac', 'slac/')
    set_default_value(config, 'template_optimized_posegraph_slac', 'optimized_posegraph_slac.json')
    set_default_value(config, 'folder_fragment', 'fragments/')
    set_default_value(config, 'subfolder_slac', 'slac/%0.3f/' % config['voxel_size'])
    set_default_value(config, 'template_fragment_posegraph', 'fragments/fragment_%03d.json')
    set_default_value(config, 'template_fragment_posegraph_optimized', 'fragments/fragment_optimized_%03d.json')
    set_default_value(config, 'template_fragment_pointcloud', 'fragments/fragment_%03d.ply')
    set_default_value(config, 'folder_scene', 'scene/')
    set_default_value(config, 'template_global_posegraph', 'scene/global_registration.json')
    set_default_value(config, 'template_global_posegraph_optimized', 'scene/global_registration_optimized.json')
    set_default_value(config, 'template_refined_posegraph', 'scene/refined_registration.json')
    set_default_value(config, 'template_refined_posegraph_optimized', 'scene/refined_registration_optimized.json')
    set_default_value(config, 'template_global_mesh', 'scene/integrated.ply')
    set_default_value(config, 'template_global_traj', 'scene/trajectory.log')
    if config['path_dataset'].endswith('.bag'):
        assert os.path.isfile(config['path_dataset']), f"File {config['path_dataset']} not found."
        print('Extracting frames from RGBD video file')
        (config['path_dataset'], config['path_intrinsic'], config['depth_scale']) = extract_rgbd_frames(config['path_dataset'])

def dataset_loader(dataset_name):
    if False:
        return 10
    print('Config file was not passed. Using deafult dataset.')
    config = {}
    if dataset_name == 'lounge':
        config = lounge_data_loader()
    elif dataset_name == 'bedroom':
        config = bedroom_data_loader()
    elif dataset_name == 'jack_jack':
        config = jackjack_data_loader()
    else:
        print('The requested dataset is not available. Available dataset options include lounge and jack_jack.')
        sys.exit(1)
    initialize_config(config)
    print('Loaded data from {}'.format(config['path_dataset']))
    return config