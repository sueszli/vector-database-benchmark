import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from os import makedirs
from os.path import exists, join, abspath
import shutil
import json
from enum import IntEnum
import sys
sys.path.append(abspath(__file__))
from realsense_helper import get_profiles
try:
    input = raw_input
except NameError:
    pass

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

def make_clean_folder(path_folder):
    if False:
        i = 10
        return i + 15
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        user_input = input('%s not empty. Overwrite? (y/n) : ' % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            exit()

def save_intrinsic_as_json(filename, frame):
    if False:
        for i in range(10):
            print('nop')
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump({'width': intrinsics.width, 'height': intrinsics.height, 'intrinsic_matrix': [intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx, intrinsics.ppy, 1]}, outfile, indent=4)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Realsense Recorder. Please select one of the optional arguments')
    parser.add_argument('--output_folder', default='../dataset/realsense/', help='set output folder')
    parser.add_argument('--record_rosbag', action='store_true', help='Recording rgbd stream into realsense.bag')
    parser.add_argument('--record_imgs', action='store_true', help='Recording save color and depth images into realsense folder')
    parser.add_argument('--playback_rosbag', action='store_true', help='Play recorded realsense.bag file')
    args = parser.parse_args()
    if sum((o is not False for o in vars(args).values())) != 2:
        parser.print_help()
        exit()
    path_output = args.output_folder
    path_depth = join(args.output_folder, 'depth')
    path_color = join(args.output_folder, 'color')
    if args.record_imgs:
        make_clean_folder(path_output)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)
    path_bag = join(args.output_folder, 'realsense.bag')
    if args.record_rosbag:
        if exists(path_bag):
            user_input = input('%s exists. Overwrite? (y/n) : ' % path_bag)
            if user_input.lower() == 'n':
                exit()
    pipeline = rs.pipeline()
    config = rs.config()
    (color_profiles, depth_profiles) = get_profiles()
    if args.record_imgs or args.record_rosbag:
        print('Using the default profiles: \n  color:{}, depth:{}'.format(color_profiles[0], depth_profiles[0]))
        (w, h, fps, fmt) = depth_profiles[0]
        config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        (w, h, fps, fmt) = color_profiles[0]
        config.enable_stream(rs.stream.color, w, h, fmt, fps)
        if args.record_rosbag:
            config.enable_record_to_file(path_bag)
    if args.playback_rosbag:
        config.enable_device_from_file(path_bag, repeat_playback=True)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    if args.record_rosbag or args.record_imgs:
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance_in_meters = 3
    clipping_distance = clipping_distance_in_meters / depth_scale
    align_to = rs.stream.color
    align = rs.align(align_to)
    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            if args.record_imgs:
                if frame_count == 0:
                    save_intrinsic_as_json(join(args.output_folder, 'camera_intrinsic.json'), color_frame)
                cv2.imwrite('%s/%06d.png' % (path_depth, frame_count), depth_image)
                cv2.imwrite('%s/%06d.jpg' % (path_color, frame_count), color_image)
                print('Saved color + depth image %06d' % frame_count)
                frame_count += 1
            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Recorder Realsense', images)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()