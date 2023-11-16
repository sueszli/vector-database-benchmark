import pyrealsense2 as rs
import numpy as np
from enum import IntEnum
from datetime import datetime
import open3d as o3d
from os.path import abspath
import sys
sys.path.append(abspath(__file__))
from realsense_helper import get_profiles

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

def get_intrinsic_matrix(frame):
    if False:
        i = 10
        return i + 15
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
    return out
if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()
    (color_profiles, depth_profiles) = get_profiles()
    print('Using the default profiles: \n  color:{}, depth:{}'.format(color_profiles[0], depth_profiles[0]))
    (w, h, fps, fmt) = depth_profiles[0]
    config.enable_stream(rs.stream.depth, w, h, fmt, fps)
    (w, h, fps, fmt) = color_profiles[0]
    config.enable_stream(rs.stream.color, w, h, fmt, fps)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance_in_meters = 3
    clipping_distance = clipping_distance_in_meters / depth_scale
    align_to = rs.stream.color
    align = rs.align(align_to)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    frame_count = 0
    try:
        while True:
            dt0 = datetime.now()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            intrinsic = o3d.camera.PinholeCameraIntrinsic(get_intrinsic_matrix(color_frame))
            if not aligned_depth_frame or not color_frame:
                continue
            depth_image = o3d.geometry.Image(np.array(aligned_depth_frame.get_data()))
            color_temp = np.asarray(color_frame.get_data())
            color_image = o3d.geometry.Image(color_temp)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale=1.0 / depth_scale, depth_trunc=clipping_distance_in_meters, convert_rgb_to_intensity=False)
            temp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            temp.transform(flip_transform)
            pcd.points = temp.points
            pcd.colors = temp.colors
            if frame_count == 0:
                vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            process_time = datetime.now() - dt0
            print('\rFPS: ' + str(1 / process_time.total_seconds()), end='')
            frame_count += 1
    finally:
        pipeline.stop()
    vis.destroy_window()