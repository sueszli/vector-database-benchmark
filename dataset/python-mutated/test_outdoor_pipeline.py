import numpy as np
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose

def test_outdoor_aug_pipeline():
    if False:
        for i in range(10):
            print('nop')
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    class_names = ['Car']
    np.random.seed(0)
    train_pipeline = [dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4), dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True), dict(type='ObjectNoise', num_try=100, translation_std=[1.0, 1.0, 0.5], global_rot_range=[0.0, 0.0], rot_range=[-0.78539816, 0.78539816]), dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5), dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]), dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range), dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range), dict(type='PointShuffle'), dict(type='DefaultFormatBundle3D', class_names=class_names), dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])]
    pipeline = Compose(train_pipeline)
    gt_bboxes_3d = LiDARInstance3DBoxes(torch.tensor([[21.6902428, -0.0406038128, -1.61906636, 1.65999997, 3.20000005, 1.61000001, 1.53999996], [7.05006886, -6.57459593, -1.60107934, 2.27999997, 12.7799997, 3.66000009, -1.54999995], [22.4698811, -6.69203758, -1.50118136, 2.31999993, 14.7299995, 3.6400001, -1.59000003], [34.8291969, -7.0905838, -1.36622977, 2.31999993, 10.04, 3.6099999, -1.61000001], [46.23946, -7.75838804, -1.32405007, 2.33999991, 12.8299999, 3.63000011, -1.63999999], [28.2966995, -0.555755794, -1.30332506, 1.47000003, 2.23000002, 1.48000002, 1.57000005], [26.6690197, 21.8230209, -1.73605704, 1.55999994, 3.48000002, 1.39999998, 1.69000006], [31.3197803, 8.16214371, -1.62177873, 1.74000001, 3.76999998, 1.48000002, -2.78999996], [43.4395561, -19.5209332, -1.20757008, 1.69000006, 4.0999999, 1.40999997, 1.53999996], [32.9882965, -3.79360509, -1.69245458, 1.74000001, 4.09000015, 1.49000001, 1.52999997], [38.546936, 8.35060215, -1.31423414, 1.59000003, 4.28000021, 1.45000005, -1.73000002], [22.2492104, -11.3536005, -1.38272512, 1.62, 3.55999994, 1.71000004, -2.48000002], [33.6115799, -19.7708054, -0.492827654, 1.64999998, 3.54999995, 1.79999995, 1.57000005], [9.85029602, -1.51294518, -1.66834795, 1.59000003, 3.17000008, 1.38999999, 0.839999974]], dtype=torch.float32))
    gt_labels_3d = np.array([0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    results = dict(pts_filename='tests/data/kitti/a.bin', ann_info=dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d), bbox3d_fields=[], img_fields=[])
    origin_center = gt_bboxes_3d.tensor[:, :3].clone()
    origin_angle = gt_bboxes_3d.tensor[:, 6].clone()
    output = pipeline(results)
    rotation_angle = output['img_metas']._data['pcd_rotation_angle']
    rotation_matrix = output['img_metas']._data['pcd_rotation']
    noise_angle = torch.tensor([0.70853819, -0.19160091, -0.71116999, 0.49571753, -0.12447527, -0.4690133, -0.34776965, -0.65692282, -0.52442831, -0.01575567, -0.61849673, 0.6572608, 0.30312288, -0.19182971])
    noise_trans = torch.tensor([[1.7641, 0.40016, 0.48937], [-1.3065, 1.6581, -0.059082], [-1.5504, 0.41732, -0.47218], [-0.52158, -1.1847, 0.48035], [-0.89637, -1.9627, 0.79241], [0.01324, -0.12194, 0.16953], [0.81798, -0.27891, 0.71578], [-0.00041733, 0.37416, 0.20478], [0.15218, -0.37413, -0.0067257], [-1.9138, -2.2855, -0.80092], [1.5933, 0.56872, -0.057244], [-1.8523, -0.71333, -0.88111], [0.52678, 0.10106, -0.19432], [-0.72449, -0.80292, -0.011334]])
    angle = -origin_angle - noise_angle + torch.tensor(rotation_angle)
    angle -= 2 * np.pi * (angle >= np.pi)
    angle += 2 * np.pi * (angle < -np.pi)
    scale = output['img_metas']._data['pcd_scale_factor']
    expected_tensor = torch.tensor([[20.6514, -8.825, -1.0816, 1.5893, 3.0637, 1.5414], [7.9374, 4.9457, -1.2008, 2.1829, 12.2357, 3.5041], [20.8115, -2.0273, -1.8893, 2.2212, 14.1026, 3.485], [32.385, -5.2135, -1.1321, 2.2212, 9.6124, 3.4562], [43.7022, -7.8316, -0.509, 2.2403, 12.2836, 3.4754], [25.33, -9.667, -1.0855, 1.4074, 2.135, 1.417], [16.5414, -29.0583, -0.9768, 1.4936, 3.3318, 1.3404], [24.6548, -18.9226, -1.3567, 1.6659, 3.6094, 1.417], [45.8403, 1.8183, -1.1626, 1.618, 3.9254, 1.3499], [30.6288, -8.4497, -1.4881, 1.6659, 3.9158, 1.4265], [32.3316, -22.4611, -1.3131, 1.5223, 4.0977, 1.3882], [22.4492, 3.2944, -2.1674, 1.551, 3.4084, 1.6372], [37.3824, 5.0472, -0.6579, 1.5797, 3.3988, 1.7233], [8.9259, -1.2578, -1.6081, 1.5223, 3.035, 1.3308]])
    expected_tensor[:, :3] = (origin_center + noise_trans) * torch.tensor([1, -1, 1]) @ rotation_matrix * scale
    expected_tensor = torch.cat([expected_tensor, angle.unsqueeze(-1)], dim=-1)
    assert torch.allclose(output['gt_bboxes_3d']._data.tensor, expected_tensor, atol=0.001)

def test_outdoor_velocity_aug_pipeline():
    if False:
        for i in range(10):
            print('nop')
    point_cloud_range = [-50, -50, -5, 50, 50, 3]
    class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
    np.random.seed(0)
    train_pipeline = [dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4), dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True), dict(type='GlobalRotScaleTrans', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05], translation_std=[0, 0, 0]), dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5), dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range), dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range), dict(type='PointShuffle'), dict(type='DefaultFormatBundle3D', class_names=class_names), dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])]
    pipeline = Compose(train_pipeline)
    gt_bboxes_3d = LiDARInstance3DBoxes(torch.tensor([[-5.2422, 40.021, -0.47643, 2.062, 4.409, 1.548, -1.488, 0.0085338, 0.044934], [-26.675, 5.595, -1.3053, 0.343, 0.458, 0.782, -4.6276, -0.00043284, -0.0018543], [-5.8098, 35.409, -0.66511, 2.396, 3.969, 1.732, -4.652, 0.0, 0.0], [-31.309, 1.0901, -1.0561, 1.944, 3.857, 1.723, -2.8143, -0.027606, -0.080573], [-45.642, 20.136, -0.024681, 1.987, 4.44, 1.942, 0.28336, 0.0, 0.0], [-5.1617, 18.305, -1.0879, 2.323, 4.851, 1.371, -1.5803, 0.0, 0.0], [-25.285, 4.1442, -1.2713, 1.755, 1.989, 2.22, -4.49, -0.031784, -0.15291], [-2.2611, 19.17, -1.1452, 0.919, 1.123, 1.931, 0.04779, 0.067684, -1.7537], [-65.878, 13.5, -0.22528, 1.82, 3.852, 1.545, -2.8757, 0.0, 0.0], [-5.449, 28.363, -0.77275, 2.236, 3.754, 1.559, -4.652, -0.0079736, 0.0077207]], dtype=torch.float32), box_dim=9)
    gt_labels_3d = np.array([0, 8, 0, 0, 0, 0, -1, 7, 0, 0])
    results = dict(pts_filename='tests/data/kitti/a.bin', ann_info=dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d), bbox3d_fields=[], img_fields=[])
    origin_center = gt_bboxes_3d.tensor[:, :3].clone()
    origin_angle = gt_bboxes_3d.tensor[:, 6].clone()
    origin_velo = gt_bboxes_3d.tensor[:, 7:9].clone()
    output = pipeline(results)
    expected_tensor = torch.tensor([[-3.7849, -41.057, -0.48668, 2.1064, 4.5039, 1.5813, -1.6919, 0.010469, -0.045533], [-27.01, -6.7551, -1.3334, 0.35038, 0.46786, 0.79883, 1.4477, -0.0005144, 0.0018758], [-4.5448, -36.372, -0.67942, 2.4476, 4.0544, 1.7693, 1.4721, 0.0, -0.0], [-31.916, -2.3379, -1.0788, 1.9858, 3.94, 1.7601, -0.36564, -0.031333, 0.081166], [-45.802, -22.34, -0.025213, 2.0298, 4.5355, 1.9838, 2.8199, 0.0, -0.0], [-4.5526, -18.887, -1.1114, 2.373, 4.9554, 1.4005, -1.5997, 0.0, -0.0], [-25.648, -5.2197, -1.2987, 1.7928, 2.0318, 2.2678, 1.31, -0.038428, 0.15485], [-1.5578, -19.657, -1.1699, 0.93878, 1.1472, 1.9726, 3.0555, 0.00045907, 1.7928], [-4.4522, -29.166, -0.78938, 2.2841, 3.8348, 1.5925, 1.4721, -0.0078371, -0.0081931]])
    rotation_angle = output['img_metas']._data['pcd_rotation_angle']
    rotation_matrix = output['img_metas']._data['pcd_rotation']
    expected_tensor[:, :3] = (origin_center @ rotation_matrix * output['img_metas']._data['pcd_scale_factor'] * torch.tensor([1, -1, 1]))[[0, 1, 2, 3, 4, 5, 6, 7, 9]]
    angle = -origin_angle - rotation_angle
    angle -= 2 * np.pi * (angle >= np.pi)
    angle += 2 * np.pi * (angle < -np.pi)
    expected_tensor[:, 6:7] = angle.unsqueeze(-1)[[0, 1, 2, 3, 4, 5, 6, 7, 9]]
    expected_tensor[:, 7:9] = (origin_velo @ rotation_matrix[:2, :2] * output['img_metas']._data['pcd_scale_factor'] * torch.tensor([1, -1]))[[0, 1, 2, 3, 4, 5, 6, 7, 9]]
    assert torch.allclose(output['gt_bboxes_3d']._data.tensor, expected_tensor, atol=0.001)