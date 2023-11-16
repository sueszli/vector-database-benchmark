import os
import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from .general_eval_dataset import read_pfm

def read_camera_parameters(filename):
    if False:
        for i in range(10):
            print('nop')
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return (intrinsics, extrinsics)

def read_img(filename):
    if False:
        print('Hello World!')
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32) / 255.0
    return np_img

def read_mask(filename):
    if False:
        return 10
    return read_img(filename) > 0.5

def save_mask(filename, mask):
    if False:
        print('Hello World!')
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

def read_pair_file(filename):
    if False:
        for i in range(10):
            print('nop')
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    if False:
        print('Hello World!')
    (width, height) = (depth_ref.shape[1], depth_ref.shape[0])
    (x_ref, y_ref) = np.meshgrid(np.arange(0, width), np.arange(0, height))
    (x_ref, y_ref) = (x_ref.reshape([-1]), y_ref.reshape([-1]))
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref), np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)), np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src), np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)), np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)
    return (depth_reprojected, x_reprojected, y_reprojected, x_src, y_src)

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    if False:
        i = 10
        return i + 15
    (width, height) = (depth_ref.shape[1], depth_ref.shape[0])
    (x_ref, y_ref) = np.meshgrid(np.arange(0, width), np.arange(0, height))
    (depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src) = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0
    return (mask, depth_reprojected, x2d_src, y2d_src)

def filter_depth(pair_folder, scan_folder, out_folder, thres_view):
    if False:
        print('Hello World!')
    pair_file = os.path.join(pair_folder, 'pair.txt')
    vertexs = []
    vertex_colors = []
    pair_data = read_pair_file(pair_file)
    for (ref_view, src_views) in pair_data:
        (ref_intrinsics, ref_extrinsics) = read_camera_parameters(os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > 0.9
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []
        geo_mask_sum = 0
        for src_view in src_views:
            (src_intrinsics, src_extrinsics) = read_camera_parameters(os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            (geo_mask, depth_reprojected, x2d_src, y2d_src) = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics, src_depth_est, src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)
        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= thres_view
        final_mask = np.logical_and(photo_mask, geo_mask)
        os.makedirs(os.path.join(out_folder, 'mask'), exist_ok=True)
        save_mask(os.path.join(out_folder, 'mask/{:0>8}_photo.png'.format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, 'mask/{:0>8}_geo.png'.format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, 'mask/{:0>8}_final.png'.format(ref_view)), final_mask)
        (height, width) = depth_est_averaged.shape[:2]
        (x, y) = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        (x, y, depth) = (x[valid_points], y[valid_points], depth_est_averaged[valid_points])
        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics), np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]
    el = PlyElement.describe(vertex_all, 'vertex')
    pcd = PlyData([el])
    return pcd

def pcd_depth_filter(scene, test_dir, save_dir, thres_view):
    if False:
        for i in range(10):
            print('nop')
    old_scene_folder = os.path.join(test_dir, scene)
    new_scene_folder = os.path.join(save_dir, scene)
    out_folder = os.path.join(save_dir, scene)
    pcd = filter_depth(old_scene_folder, new_scene_folder, out_folder, thres_view)
    return pcd