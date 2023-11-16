import numpy as np
import pytest
import torch
from mmdet3d.core.points import BasePoints, CameraPoints, DepthPoints, LiDARPoints

def test_base_points():
    if False:
        i = 10
        return i + 15
    empty_boxes = []
    points = BasePoints(empty_boxes)
    assert points.tensor.shape[0] == 0
    assert points.tensor.shape[1] == 3
    points_np = np.array([[-5.24223238, 40.0209696, 0.297570381], [-26.6751588, 5.59499564, -0.91434586], [-5.80979675, 35.4092357, 0.200889888], [-31.3086877, 1.09007628, -0.194612112]], dtype=np.float32)
    base_points = BasePoints(points_np, points_dim=3)
    assert base_points.tensor.shape[0] == 4
    points_np = np.array([[-5.24223238, 40.0209696, 0.297570381, 0.6666, 0.1956, 0.4974, 0.9409], [-26.6751588, 5.59499564, -0.91434586, 0.1502, 0.3707, 0.1086, 0.6297], [-5.80979675, 35.4092357, 0.200889888, 0.6565, 0.6248, 0.6954, 0.2538], [-31.3086877, 1.09007628, -0.194612112, 0.2803, 0.0258, 0.4896, 0.3269]], dtype=np.float32)
    base_points = BasePoints(points_np, points_dim=7, attribute_dims=dict(color=[3, 4, 5], height=6))
    expected_tensor = torch.tensor([[-5.24223238, 40.0209696, 0.297570381, 0.6666, 0.1956, 0.4974, 0.9409], [-26.6751588, 5.59499564, -0.91434586, 0.1502, 0.3707, 0.1086, 0.6297], [-5.80979675, 35.4092357, 0.200889888, 0.6565, 0.6248, 0.6954, 0.2538], [-31.3086877, 1.09007628, -0.194612112, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, base_points.tensor)
    assert torch.allclose(expected_tensor[:, :2], base_points.bev)
    assert torch.allclose(expected_tensor[:, :3], base_points.coord)
    assert torch.allclose(expected_tensor[:, 3:6], base_points.color)
    assert torch.allclose(expected_tensor[:, 6], base_points.height)
    new_base_points = base_points.clone()
    assert torch.allclose(new_base_points.tensor, base_points.tensor)
    new_base_points.shuffle()
    assert new_base_points.tensor.shape == torch.Size([4, 7])
    rot_mat = torch.tensor([[0.93629336, -0.27509585, 0.21835066], [0.28962948, 0.95642509, -0.03695701], [-0.19866933, 0.0978434, 0.97517033]])
    base_points.rotate(rot_mat)
    expected_tensor = torch.tensor([[6.6239, 39.748, -2.3335, 0.6666, 0.1956, 0.4974, 0.9409], [-23.174, 12.6, -6.923, 0.1502, 0.3707, 0.1086, 0.6297], [4.776, 35.484, -2.3813, 0.6565, 0.6248, 0.6954, 0.2538], [-28.96, 9.6364, -7.0663, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, base_points.tensor, 0.001)
    new_base_points = base_points.clone()
    new_base_points.rotate(0.1, axis=2)
    expected_tensor = torch.tensor([[2.6226, 40.211, -2.3335, 0.6666, 0.1956, 0.4974, 0.9409], [-24.316, 10.224, -6.923, 0.1502, 0.3707, 0.1086, 0.6297], [1.2096, 35.784, -2.3813, 0.6565, 0.6248, 0.6954, 0.2538], [-29.777, 6.6971, -7.0663, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, new_base_points.tensor, 0.001)
    translation_vector = torch.tensor([0.93629336, -0.27509585, 0.21835066])
    base_points.translate(translation_vector)
    expected_tensor = torch.tensor([[7.5602, 39.473, -2.1152, 0.6666, 0.1956, 0.4974, 0.9409], [-22.237, 12.325, -6.7046, 0.1502, 0.3707, 0.1086, 0.6297], [5.7123, 35.209, -2.1629, 0.6565, 0.6248, 0.6954, 0.2538], [-28.023, 9.3613, -6.848, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, base_points.tensor, 0.0001)
    point_range = [-10, -40, -10, 10, 40, 10]
    in_range_flags = base_points.in_range_3d(point_range)
    expected_flags = torch.tensor([True, False, True, False])
    assert torch.all(in_range_flags == expected_flags)
    base_points.scale(1.2)
    expected_tensor = torch.tensor([[9.0722, 47.368, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [-26.685, 14.79, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [6.8547, 42.251, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [-33.628, 11.234, -8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, base_points.tensor, 0.001)
    expected_tensor = torch.tensor([[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297]])
    assert torch.allclose(expected_tensor, base_points[1].tensor, 0.0001)
    expected_tensor = torch.tensor([[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, base_points[1:3].tensor, 0.0001)
    mask = torch.tensor([True, False, True, False])
    expected_tensor = torch.tensor([[9.0722, 47.3678, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, base_points[mask].tensor, 0.0001)
    expected_tensor = torch.tensor([[0.6666], [0.1502], [0.6565], [0.2803]])
    assert torch.allclose(expected_tensor, base_points[:, 3].tensor, 0.0001)
    assert len(base_points) == 4
    expected_repr = 'BasePoints(\n    tensor([[ 9.0722e+00,  4.7368e+01, -2.5382e+00,  6.6660e-01,  1.9560e-01,\n          4.9740e-01,  9.4090e-01],\n        [-2.6685e+01,  1.4790e+01, -8.0455e+00,  1.5020e-01,  3.7070e-01,\n          1.0860e-01,  6.2970e-01],\n        [ 6.8547e+00,  4.2251e+01, -2.5955e+00,  6.5650e-01,  6.2480e-01,\n          6.9540e-01,  2.5380e-01],\n        [-3.3628e+01,  1.1234e+01, -8.2176e+00,  2.8030e-01,  2.5800e-02,\n          4.8960e-01,  3.2690e-01]]))'
    assert expected_repr == str(base_points)
    base_points_clone = base_points.clone()
    cat_points = BasePoints.cat([base_points, base_points_clone])
    assert torch.allclose(cat_points.tensor[:len(base_points)], base_points.tensor)
    for (i, point) in enumerate(base_points):
        assert torch.allclose(point, base_points.tensor[i])
    new_points = base_points.new_point([[1, 2, 3, 4, 5, 6, 7]])
    assert torch.allclose(new_points.tensor, torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=base_points.tensor.dtype))
    base_points = BasePoints(points_np, points_dim=7, attribute_dims=dict(height=3, color=[4, 5, 6]))
    assert torch.all(base_points[:, 3:].tensor == torch.tensor(points_np[:, 3:]))
    base_points = BasePoints(points_np[:, :3])
    assert base_points.attribute_dims is None
    base_points.height = points_np[:, 3]
    assert base_points.attribute_dims == dict(height=3)
    base_points.color = points_np[:, 4:]
    assert base_points.attribute_dims == dict(height=3, color=[4, 5, 6])
    assert torch.allclose(base_points.height, torch.tensor([0.6666, 0.1502, 0.6565, 0.2803]))
    assert torch.allclose(base_points.color, torch.tensor([[0.1956, 0.4974, 0.9409], [0.3707, 0.1086, 0.6297], [0.6248, 0.6954, 0.2538], [0.0258, 0.4896, 0.3269]]))
    with pytest.raises(ValueError):
        base_points.coord = np.random.rand(5, 3)
    with pytest.raises(ValueError):
        base_points.height = np.random.rand(3)
    with pytest.raises(ValueError):
        base_points.color = np.random.rand(4, 2)
    base_points.coord = points_np[:, [1, 2, 3]]
    base_points.height = points_np[:, 0]
    base_points.color = points_np[:, [4, 5, 6]]
    assert np.allclose(base_points.coord, points_np[:, 1:4])
    assert np.allclose(base_points.height, points_np[:, 0])
    assert np.allclose(base_points.color, points_np[:, 4:])

def test_cam_points():
    if False:
        for i in range(10):
            print('nop')
    empty_boxes = []
    points = CameraPoints(empty_boxes)
    assert points.tensor.shape[0] == 0
    assert points.tensor.shape[1] == 3
    points_np = np.array([[-5.24223238, 40.0209696, 0.297570381], [-26.6751588, 5.59499564, -0.91434586], [-5.80979675, 35.4092357, 0.200889888], [-31.3086877, 1.09007628, -0.194612112]], dtype=np.float32)
    cam_points = CameraPoints(points_np, points_dim=3)
    assert cam_points.tensor.shape[0] == 4
    points_np = np.array([[-5.24223238, 40.0209696, 0.297570381, 0.6666, 0.1956, 0.4974, 0.9409], [-26.6751588, 5.59499564, -0.91434586, 0.1502, 0.3707, 0.1086, 0.6297], [-5.80979675, 35.4092357, 0.200889888, 0.6565, 0.6248, 0.6954, 0.2538], [-31.3086877, 1.09007628, -0.194612112, 0.2803, 0.0258, 0.4896, 0.3269]], dtype=np.float32)
    cam_points = CameraPoints(points_np, points_dim=7, attribute_dims=dict(color=[3, 4, 5], height=6))
    expected_tensor = torch.tensor([[-5.24223238, 40.0209696, 0.297570381, 0.6666, 0.1956, 0.4974, 0.9409], [-26.6751588, 5.59499564, -0.91434586, 0.1502, 0.3707, 0.1086, 0.6297], [-5.80979675, 35.4092357, 0.200889888, 0.6565, 0.6248, 0.6954, 0.2538], [-31.3086877, 1.09007628, -0.194612112, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, cam_points.tensor)
    assert torch.allclose(expected_tensor[:, [0, 2]], cam_points.bev)
    assert torch.allclose(expected_tensor[:, :3], cam_points.coord)
    assert torch.allclose(expected_tensor[:, 3:6], cam_points.color)
    assert torch.allclose(expected_tensor[:, 6], cam_points.height)
    new_cam_points = cam_points.clone()
    assert torch.allclose(new_cam_points.tensor, cam_points.tensor)
    new_cam_points.shuffle()
    assert new_cam_points.tensor.shape == torch.Size([4, 7])
    rot_mat = torch.tensor([[0.93629336, -0.27509585, 0.21835066], [0.28962948, 0.95642509, -0.03695701], [-0.19866933, 0.0978434, 0.97517033]])
    cam_points.rotate(rot_mat)
    expected_tensor = torch.tensor([[6.6239, 39.748, -2.3335, 0.6666, 0.1956, 0.4974, 0.9409], [-23.174, 12.6, -6.923, 0.1502, 0.3707, 0.1086, 0.6297], [4.776, 35.484, -2.3813, 0.6565, 0.6248, 0.6954, 0.2538], [-28.96, 9.6364, -7.0663, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, cam_points.tensor, 0.001)
    new_cam_points = cam_points.clone()
    new_cam_points.rotate(0.1, axis=2)
    expected_tensor = torch.tensor([[2.6226, 40.211, -2.3335, 0.6666, 0.1956, 0.4974, 0.9409], [-24.316, 10.224, -6.923, 0.1502, 0.3707, 0.1086, 0.6297], [1.2096, 35.784, -2.3813, 0.6565, 0.6248, 0.6954, 0.2538], [-29.777, 6.6971, -7.0663, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, new_cam_points.tensor, 0.001)
    translation_vector = torch.tensor([0.93629336, -0.27509585, 0.21835066])
    cam_points.translate(translation_vector)
    expected_tensor = torch.tensor([[7.5602, 39.473, -2.1152, 0.6666, 0.1956, 0.4974, 0.9409], [-22.237, 12.325, -6.7046, 0.1502, 0.3707, 0.1086, 0.6297], [5.7123, 35.209, -2.1629, 0.6565, 0.6248, 0.6954, 0.2538], [-28.023, 9.3613, -6.848, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, cam_points.tensor, 0.0001)
    point_range = [-10, -40, -10, 10, 40, 10]
    in_range_flags = cam_points.in_range_3d(point_range)
    expected_flags = torch.tensor([True, False, True, False])
    assert torch.all(in_range_flags == expected_flags)
    cam_points.scale(1.2)
    expected_tensor = torch.tensor([[9.0722, 47.368, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [-26.685, 14.79, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [6.8547, 42.251, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [-33.628, 11.234, -8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, cam_points.tensor, 0.001)
    expected_tensor = torch.tensor([[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297]])
    assert torch.allclose(expected_tensor, cam_points[1].tensor, 0.0001)
    expected_tensor = torch.tensor([[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, cam_points[1:3].tensor, 0.0001)
    mask = torch.tensor([True, False, True, False])
    expected_tensor = torch.tensor([[9.0722, 47.3678, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, cam_points[mask].tensor, 0.0001)
    expected_tensor = torch.tensor([[0.6666], [0.1502], [0.6565], [0.2803]])
    assert torch.allclose(expected_tensor, cam_points[:, 3].tensor, 0.0001)
    assert len(cam_points) == 4
    expected_repr = 'CameraPoints(\n    tensor([[ 9.0722e+00,  4.7368e+01, -2.5382e+00,  6.6660e-01,  1.9560e-01,\n          4.9740e-01,  9.4090e-01],\n        [-2.6685e+01,  1.4790e+01, -8.0455e+00,  1.5020e-01,  3.7070e-01,\n          1.0860e-01,  6.2970e-01],\n        [ 6.8547e+00,  4.2251e+01, -2.5955e+00,  6.5650e-01,  6.2480e-01,\n          6.9540e-01,  2.5380e-01],\n        [-3.3628e+01,  1.1234e+01, -8.2176e+00,  2.8030e-01,  2.5800e-02,\n          4.8960e-01,  3.2690e-01]]))'
    assert expected_repr == str(cam_points)
    cam_points_clone = cam_points.clone()
    cat_points = CameraPoints.cat([cam_points, cam_points_clone])
    assert torch.allclose(cat_points.tensor[:len(cam_points)], cam_points.tensor)
    for (i, point) in enumerate(cam_points):
        assert torch.allclose(point, cam_points.tensor[i])
    new_points = cam_points.new_point([[1, 2, 3, 4, 5, 6, 7]])
    assert torch.allclose(new_points.tensor, torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=cam_points.tensor.dtype))
    point_bev_range = [-10, -10, 10, 10]
    in_range_flags = cam_points.in_range_bev(point_bev_range)
    expected_flags = torch.tensor([True, False, True, False])
    assert torch.all(in_range_flags == expected_flags)
    cam_points.flip(bev_direction='horizontal')
    expected_tensor = torch.tensor([[-9.0722, 47.368, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [26.685, 14.79, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [-6.8547, 42.251, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [33.628, 11.234, -8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, cam_points.tensor, 0.0001)
    cam_points.flip(bev_direction='vertical')
    expected_tensor = torch.tensor([[-9.0722, 47.368, 2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [26.685, 14.79, 8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [-6.8547, 42.251, 2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [33.628, 11.234, 8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, cam_points.tensor, 0.0001)

def test_lidar_points():
    if False:
        print('Hello World!')
    empty_boxes = []
    points = LiDARPoints(empty_boxes)
    assert points.tensor.shape[0] == 0
    assert points.tensor.shape[1] == 3
    points_np = np.array([[-5.24223238, 40.0209696, 0.297570381], [-26.6751588, 5.59499564, -0.91434586], [-5.80979675, 35.4092357, 0.200889888], [-31.3086877, 1.09007628, -0.194612112]], dtype=np.float32)
    lidar_points = LiDARPoints(points_np, points_dim=3)
    assert lidar_points.tensor.shape[0] == 4
    points_np = np.array([[-5.24223238, 40.0209696, 0.297570381, 0.6666, 0.1956, 0.4974, 0.9409], [-26.6751588, 5.59499564, -0.91434586, 0.1502, 0.3707, 0.1086, 0.6297], [-5.80979675, 35.4092357, 0.200889888, 0.6565, 0.6248, 0.6954, 0.2538], [-31.3086877, 1.09007628, -0.194612112, 0.2803, 0.0258, 0.4896, 0.3269]], dtype=np.float32)
    lidar_points = LiDARPoints(points_np, points_dim=7, attribute_dims=dict(color=[3, 4, 5], height=6))
    expected_tensor = torch.tensor([[-5.24223238, 40.0209696, 0.297570381, 0.6666, 0.1956, 0.4974, 0.9409], [-26.6751588, 5.59499564, -0.91434586, 0.1502, 0.3707, 0.1086, 0.6297], [-5.80979675, 35.4092357, 0.200889888, 0.6565, 0.6248, 0.6954, 0.2538], [-31.3086877, 1.09007628, -0.194612112, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, lidar_points.tensor)
    assert torch.allclose(expected_tensor[:, :2], lidar_points.bev)
    assert torch.allclose(expected_tensor[:, :3], lidar_points.coord)
    assert torch.allclose(expected_tensor[:, 3:6], lidar_points.color)
    assert torch.allclose(expected_tensor[:, 6], lidar_points.height)
    new_lidar_points = lidar_points.clone()
    assert torch.allclose(new_lidar_points.tensor, lidar_points.tensor)
    new_lidar_points.shuffle()
    assert new_lidar_points.tensor.shape == torch.Size([4, 7])
    rot_mat = torch.tensor([[0.93629336, -0.27509585, 0.21835066], [0.28962948, 0.95642509, -0.03695701], [-0.19866933, 0.0978434, 0.97517033]])
    lidar_points.rotate(rot_mat)
    expected_tensor = torch.tensor([[6.6239, 39.748, -2.3335, 0.6666, 0.1956, 0.4974, 0.9409], [-23.174, 12.6, -6.923, 0.1502, 0.3707, 0.1086, 0.6297], [4.776, 35.484, -2.3813, 0.6565, 0.6248, 0.6954, 0.2538], [-28.96, 9.6364, -7.0663, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, lidar_points.tensor, 0.001)
    new_lidar_points = lidar_points.clone()
    new_lidar_points.rotate(0.1, axis=2)
    expected_tensor = torch.tensor([[2.6226, 40.211, -2.3335, 0.6666, 0.1956, 0.4974, 0.9409], [-24.316, 10.224, -6.923, 0.1502, 0.3707, 0.1086, 0.6297], [1.2096, 35.784, -2.3813, 0.6565, 0.6248, 0.6954, 0.2538], [-29.777, 6.6971, -7.0663, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, new_lidar_points.tensor, 0.001)
    translation_vector = torch.tensor([0.93629336, -0.27509585, 0.21835066])
    lidar_points.translate(translation_vector)
    expected_tensor = torch.tensor([[7.5602, 39.473, -2.1152, 0.6666, 0.1956, 0.4974, 0.9409], [-22.237, 12.325, -6.7046, 0.1502, 0.3707, 0.1086, 0.6297], [5.7123, 35.209, -2.1629, 0.6565, 0.6248, 0.6954, 0.2538], [-28.023, 9.3613, -6.848, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, lidar_points.tensor, 0.0001)
    point_range = [-10, -40, -10, 10, 40, 10]
    in_range_flags = lidar_points.in_range_3d(point_range)
    expected_flags = torch.tensor([True, False, True, False])
    assert torch.all(in_range_flags == expected_flags)
    lidar_points.scale(1.2)
    expected_tensor = torch.tensor([[9.0722, 47.368, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [-26.685, 14.79, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [6.8547, 42.251, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [-33.628, 11.234, -8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, lidar_points.tensor, 0.001)
    expected_tensor = torch.tensor([[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297]])
    assert torch.allclose(expected_tensor, lidar_points[1].tensor, 0.0001)
    expected_tensor = torch.tensor([[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, lidar_points[1:3].tensor, 0.0001)
    mask = torch.tensor([True, False, True, False])
    expected_tensor = torch.tensor([[9.0722, 47.3678, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, lidar_points[mask].tensor, 0.0001)
    expected_tensor = torch.tensor([[0.6666], [0.1502], [0.6565], [0.2803]])
    assert torch.allclose(expected_tensor, lidar_points[:, 3].tensor, 0.0001)
    assert len(lidar_points) == 4
    expected_repr = 'LiDARPoints(\n    tensor([[ 9.0722e+00,  4.7368e+01, -2.5382e+00,  6.6660e-01,  1.9560e-01,\n          4.9740e-01,  9.4090e-01],\n        [-2.6685e+01,  1.4790e+01, -8.0455e+00,  1.5020e-01,  3.7070e-01,\n          1.0860e-01,  6.2970e-01],\n        [ 6.8547e+00,  4.2251e+01, -2.5955e+00,  6.5650e-01,  6.2480e-01,\n          6.9540e-01,  2.5380e-01],\n        [-3.3628e+01,  1.1234e+01, -8.2176e+00,  2.8030e-01,  2.5800e-02,\n          4.8960e-01,  3.2690e-01]]))'
    assert expected_repr == str(lidar_points)
    lidar_points_clone = lidar_points.clone()
    cat_points = LiDARPoints.cat([lidar_points, lidar_points_clone])
    assert torch.allclose(cat_points.tensor[:len(lidar_points)], lidar_points.tensor)
    for (i, point) in enumerate(lidar_points):
        assert torch.allclose(point, lidar_points.tensor[i])
    new_points = lidar_points.new_point([[1, 2, 3, 4, 5, 6, 7]])
    assert torch.allclose(new_points.tensor, torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=lidar_points.tensor.dtype))
    point_bev_range = [-30, -40, 30, 40]
    in_range_flags = lidar_points.in_range_bev(point_bev_range)
    expected_flags = torch.tensor([False, True, False, False])
    assert torch.all(in_range_flags == expected_flags)
    lidar_points.flip(bev_direction='horizontal')
    expected_tensor = torch.tensor([[9.0722, -47.368, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [-26.685, -14.79, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [6.8547, -42.251, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [-33.628, -11.234, -8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, lidar_points.tensor, 0.0001)
    lidar_points.flip(bev_direction='vertical')
    expected_tensor = torch.tensor([[-9.0722, -47.368, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [26.685, -14.79, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [-6.8547, -42.251, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [33.628, -11.234, -8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, lidar_points.tensor, 0.0001)

def test_depth_points():
    if False:
        while True:
            i = 10
    empty_boxes = []
    points = DepthPoints(empty_boxes)
    assert points.tensor.shape[0] == 0
    assert points.tensor.shape[1] == 3
    points_np = np.array([[-5.24223238, 40.0209696, 0.297570381], [-26.6751588, 5.59499564, -0.91434586], [-5.80979675, 35.4092357, 0.200889888], [-31.3086877, 1.09007628, -0.194612112]], dtype=np.float32)
    depth_points = DepthPoints(points_np, points_dim=3)
    assert depth_points.tensor.shape[0] == 4
    points_np = np.array([[-5.24223238, 40.0209696, 0.297570381, 0.6666, 0.1956, 0.4974, 0.9409], [-26.6751588, 5.59499564, -0.91434586, 0.1502, 0.3707, 0.1086, 0.6297], [-5.80979675, 35.4092357, 0.200889888, 0.6565, 0.6248, 0.6954, 0.2538], [-31.3086877, 1.09007628, -0.194612112, 0.2803, 0.0258, 0.4896, 0.3269]], dtype=np.float32)
    depth_points = DepthPoints(points_np, points_dim=7, attribute_dims=dict(color=[3, 4, 5], height=6))
    expected_tensor = torch.tensor([[-5.24223238, 40.0209696, 0.297570381, 0.6666, 0.1956, 0.4974, 0.9409], [-26.6751588, 5.59499564, -0.91434586, 0.1502, 0.3707, 0.1086, 0.6297], [-5.80979675, 35.4092357, 0.200889888, 0.6565, 0.6248, 0.6954, 0.2538], [-31.3086877, 1.09007628, -0.194612112, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, depth_points.tensor)
    assert torch.allclose(expected_tensor[:, :2], depth_points.bev)
    assert torch.allclose(expected_tensor[:, :3], depth_points.coord)
    assert torch.allclose(expected_tensor[:, 3:6], depth_points.color)
    assert torch.allclose(expected_tensor[:, 6], depth_points.height)
    new_depth_points = depth_points.clone()
    assert torch.allclose(new_depth_points.tensor, depth_points.tensor)
    new_depth_points.shuffle()
    assert new_depth_points.tensor.shape == torch.Size([4, 7])
    rot_mat = torch.tensor([[0.93629336, -0.27509585, 0.21835066], [0.28962948, 0.95642509, -0.03695701], [-0.19866933, 0.0978434, 0.97517033]])
    depth_points.rotate(rot_mat)
    expected_tensor = torch.tensor([[6.6239, 39.748, -2.3335, 0.6666, 0.1956, 0.4974, 0.9409], [-23.174, 12.6, -6.923, 0.1502, 0.3707, 0.1086, 0.6297], [4.776, 35.484, -2.3813, 0.6565, 0.6248, 0.6954, 0.2538], [-28.96, 9.6364, -7.0663, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, depth_points.tensor, 0.001)
    new_depth_points = depth_points.clone()
    new_depth_points.rotate(0.1, axis=2)
    expected_tensor = torch.tensor([[2.6226, 40.211, -2.3335, 0.6666, 0.1956, 0.4974, 0.9409], [-24.316, 10.224, -6.923, 0.1502, 0.3707, 0.1086, 0.6297], [1.2096, 35.784, -2.3813, 0.6565, 0.6248, 0.6954, 0.2538], [-29.777, 6.6971, -7.0663, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, new_depth_points.tensor, 0.001)
    translation_vector = torch.tensor([0.93629336, -0.27509585, 0.21835066])
    depth_points.translate(translation_vector)
    expected_tensor = torch.tensor([[7.5602, 39.473, -2.1152, 0.6666, 0.1956, 0.4974, 0.9409], [-22.237, 12.325, -6.7046, 0.1502, 0.3707, 0.1086, 0.6297], [5.7123, 35.209, -2.1629, 0.6565, 0.6248, 0.6954, 0.2538], [-28.023, 9.3613, -6.848, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, depth_points.tensor, 0.0001)
    point_range = [-10, -40, -10, 10, 40, 10]
    in_range_flags = depth_points.in_range_3d(point_range)
    expected_flags = torch.tensor([True, False, True, False])
    assert torch.all(in_range_flags == expected_flags)
    depth_points.scale(1.2)
    expected_tensor = torch.tensor([[9.0722, 47.368, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [-26.685, 14.79, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [6.8547, 42.251, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [-33.628, 11.234, -8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, depth_points.tensor, 0.001)
    expected_tensor = torch.tensor([[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297]])
    assert torch.allclose(expected_tensor, depth_points[1].tensor, 0.0001)
    expected_tensor = torch.tensor([[-26.6848, 14.7898, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, depth_points[1:3].tensor, 0.0001)
    mask = torch.tensor([True, False, True, False])
    expected_tensor = torch.tensor([[9.0722, 47.3678, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [6.8547, 42.2509, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538]])
    assert torch.allclose(expected_tensor, depth_points[mask].tensor, 0.0001)
    expected_tensor = torch.tensor([[0.6666], [0.1502], [0.6565], [0.2803]])
    assert torch.allclose(expected_tensor, depth_points[:, 3].tensor, 0.0001)
    assert len(depth_points) == 4
    expected_repr = 'DepthPoints(\n    tensor([[ 9.0722e+00,  4.7368e+01, -2.5382e+00,  6.6660e-01,  1.9560e-01,\n          4.9740e-01,  9.4090e-01],\n        [-2.6685e+01,  1.4790e+01, -8.0455e+00,  1.5020e-01,  3.7070e-01,\n          1.0860e-01,  6.2970e-01],\n        [ 6.8547e+00,  4.2251e+01, -2.5955e+00,  6.5650e-01,  6.2480e-01,\n          6.9540e-01,  2.5380e-01],\n        [-3.3628e+01,  1.1234e+01, -8.2176e+00,  2.8030e-01,  2.5800e-02,\n          4.8960e-01,  3.2690e-01]]))'
    assert expected_repr == str(depth_points)
    depth_points_clone = depth_points.clone()
    cat_points = DepthPoints.cat([depth_points, depth_points_clone])
    assert torch.allclose(cat_points.tensor[:len(depth_points)], depth_points.tensor)
    for (i, point) in enumerate(depth_points):
        assert torch.allclose(point, depth_points.tensor[i])
    new_points = depth_points.new_point([[1, 2, 3, 4, 5, 6, 7]])
    assert torch.allclose(new_points.tensor, torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=depth_points.tensor.dtype))
    point_bev_range = [-30, -40, 30, 40]
    in_range_flags = depth_points.in_range_bev(point_bev_range)
    expected_flags = torch.tensor([False, True, False, False])
    assert torch.all(in_range_flags == expected_flags)
    depth_points.flip(bev_direction='horizontal')
    expected_tensor = torch.tensor([[-9.0722, 47.368, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [26.685, 14.79, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [-6.8547, 42.251, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [33.628, 11.234, -8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, depth_points.tensor, 0.0001)
    depth_points.flip(bev_direction='vertical')
    expected_tensor = torch.tensor([[-9.0722, -47.368, -2.5382, 0.6666, 0.1956, 0.4974, 0.9409], [26.685, -14.79, -8.0455, 0.1502, 0.3707, 0.1086, 0.6297], [-6.8547, -42.251, -2.5955, 0.6565, 0.6248, 0.6954, 0.2538], [33.628, -11.234, -8.2176, 0.2803, 0.0258, 0.4896, 0.3269]])
    assert torch.allclose(expected_tensor, depth_points.tensor, 0.0001)