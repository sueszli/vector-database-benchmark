"""Tests coords transformation in fusion modules.

CommandLine:
    pytest tests/test_models/test_fusion/test_fusion_coord_trans.py
"""
import torch
from mmdet3d.models.fusion_layers import apply_3d_transformation

def test_coords_transformation():
    if False:
        return 10
    'Test the transformation of 3d coords.'
    img_meta = {'pcd_scale_factor': 1.2311, 'pcd_rotation': [[0.8660254, 0.5, 0], [-0.5, 0.8660254, 0], [0, 0, 1.0]], 'pcd_trans': [0.01111, -0.00888, 0.0], 'pcd_horizontal_flip': True, 'transformation_3d_flow': ['HF', 'R', 'S', 'T']}
    pcd = torch.tensor([[-5.2422, -0.29757, 40.021], [-0.91435, 26.675, -5.595], [0.20089, 5.8098, -35.409], [-0.19461, 31.309, -1.0901]])
    pcd_transformed = apply_3d_transformation(pcd, 'DEPTH', img_meta, reverse=False)
    expected_tensor = torch.tensor([[5.78332345, 2.900697, 49.2698531], [-15.433839, 28.99385, -6.8880045], [-3.77929405, 6.061661, -43.5920199], [-19.053658, 33.491436, -1.34202211]])
    assert torch.allclose(expected_tensor, pcd_transformed, 0.0001)
    img_meta = {'pcd_scale_factor': 0.707106781, 'pcd_rotation': [[0.707106781, 0.707106781, 0.0], [-0.707106781, 0.707106781, 0.0], [0.0, 0.0, 1.0]], 'pcd_trans': [0.0, 0.0, 0.0], 'pcd_horizontal_flip': False, 'transformation_3d_flow': ['HF', 'R', 'S', 'T']}
    pcd = torch.tensor([[-5.2422, -0.29757, 40.021], [-91.435, 26.675, -5.595], [6.061661, -0.0, -100.0]])
    pcd_transformed = apply_3d_transformation(pcd, 'DEPTH', img_meta, reverse=True)
    expected_tensor = torch.tensor([[-5.53977, 4.94463, 56.5982409], [-64.76, 118.11, -7.91252488], [6.061661, -6.061661, -141.421356]])
    assert torch.allclose(expected_tensor, pcd_transformed, 0.0001)
    img_meta = {'pcd_scale_factor': 1.0 / 0.707106781, 'pcd_rotation': [[0.707106781, 0.0, 0.707106781], [0.0, 1.0, 0.0], [-0.707106781, 0.0, 0.707106781]], 'pcd_trans': [1.0, -1.0, 0.0], 'pcd_horizontal_flip': True, 'transformation_3d_flow': ['HF', 'S', 'R', 'T']}
    pcd = torch.tensor([[-5.2422, 40.021, -0.29757], [-91.435, -5.595, 26.675], [6.061661, -100.0, -0.0]])
    pcd_transformed = apply_3d_transformation(pcd, 'CAMERA', img_meta, reverse=False)
    expected_tensor = torch.tensor([[6.53977, 55.5982409, 4.94463], [65.76, -8.91252488, 118.11], [-5.061661, -142.421356, -6.061661]])
    assert torch.allclose(expected_tensor, pcd_transformed, 0.0001)
    img_meta = {'pcd_vertical_flip': True, 'transformation_3d_flow': ['VF']}
    pcd_transformed = apply_3d_transformation(pcd, 'CAMERA', img_meta, reverse=True)
    expected_tensor = torch.tensor([[-5.2422, 40.021, 0.29757], [-91.435, -5.595, -26.675], [6.061661, -100.0, 0.0]])
    assert torch.allclose(expected_tensor, pcd_transformed, 0.0001)
    img_meta = {'pcd_vertical_flip': True, 'pcd_horizontal_flip': True, 'transformation_3d_flow': ['VF', 'HF']}
    pcd_transformed = apply_3d_transformation(pcd, 'DEPTH', img_meta, reverse=False)
    expected_tensor = torch.tensor([[5.2422, -40.021, -0.29757], [91.435, 5.595, 26.675], [-6.061661, 100.0, 0.0]])
    assert torch.allclose(expected_tensor, pcd_transformed, 0.0001)
    img_meta = {'pcd_vertical_flip': True, 'pcd_horizontal_flip': True, 'transformation_3d_flow': ['VF', 'HF']}
    pcd_transformed = apply_3d_transformation(pcd, 'LIDAR', img_meta, reverse=True)
    expected_tensor = torch.tensor([[5.2422, -40.021, -0.29757], [91.435, 5.595, 26.675], [-6.061661, 100.0, 0.0]])
    assert torch.allclose(expected_tensor, pcd_transformed, 0.0001)