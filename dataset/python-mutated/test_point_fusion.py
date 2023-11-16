"""Tests the core function of point fusion.

CommandLine:
    pytest tests/test_models/test_fusion/test_point_fusion.py
"""
import torch
from mmdet3d.models.fusion_layers import PointFusion

def test_sample_single():
    if False:
        for i in range(10):
            print('nop')
    lidar2img = torch.tensor([[602.94, -707.91, -12.275, -170.94], [176.78, 8.8088, -707.94, -102.57], [0.99998, -0.0015283, -0.0052907, -0.32757], [0.0, 0.0, 0.0, 1.0]])
    img_meta = {'transformation_3d_flow': ['R', 'S', 'T', 'HF'], 'input_shape': [370, 1224], 'img_shape': [370, 1224], 'lidar2img': lidar2img}
    fuse = PointFusion(1, 1, 1, 1)
    img_feat = torch.arange(370 * 1224)[None, ...].view(370, 1224)[None, None, ...].float() / (370 * 1224)
    pts = torch.tensor([[8.356, -4.312, -0.445], [11.777, -6.724, -0.564], [6.453, 2.53, -1.612], [6.227, -3.839, -0.563]])
    out = fuse.sample_single(img_feat, pts, img_meta)
    expected_tensor = torch.tensor([0.5560822, 0.5476625, 0.9687978, 0.6241757])
    assert torch.allclose(expected_tensor, out, 0.0001)
    pcd_rotation = torch.tensor([[0.8660254, 0.5, 0], [-0.5, 0.8660254, 0], [0, 0, 1.0]])
    pcd_scale_factor = 1.111
    pcd_trans = torch.tensor([1.0, -1.0, 0.5])
    pts = pts @ pcd_rotation
    pts *= pcd_scale_factor
    pts += pcd_trans
    pts[:, 1] = -pts[:, 1]
    img_meta.update({'pcd_scale_factor': pcd_scale_factor, 'pcd_rotation': pcd_rotation, 'pcd_trans': pcd_trans, 'pcd_horizontal_flip': True})
    out = fuse.sample_single(img_feat, pts, img_meta)
    expected_tensor = torch.tensor([0.5560822, 0.5476625, 0.9687978, 0.6241757])
    assert torch.allclose(expected_tensor, out, 0.0001)