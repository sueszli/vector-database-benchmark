from __future__ import annotations
from typing import Any
import torch
from kornia.core import Tensor
from kornia.utils import create_meshgrid
from .geometry import warp_kpts

@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt: Tensor, mask: Tensor) -> Tensor:
    if False:
        return 10
    'For megadepth dataset, zero-padding exists in images.'
    (n, h, w) = mask.shape
    mask = mask.reshape(n, h * w).unsqueeze(-1).repeat(1, 1, 2)
    grid_pt[~mask.bool()] = 0
    return grid_pt

@torch.no_grad()
def spvs_coarse(data: dict[str, Any], config: dict[str, Any]) -> None:
    if False:
        print('Hello World!')
    '\n    Update:\n        data (dict): {\n            "conf_matrix_gt": [N, hw0, hw1],\n            \'spv_b_ids\': [M]\n            \'spv_i_ids\': [M]\n            \'spv_j_ids\': [M]\n            \'spv_w_pt0_i\': [N, hw0, 2], in original image resolution\n            \'spv_pt1_i\': [N, hw1, 2], in original image resolution\n        }\n\n    NOTE:\n        - for scannet dataset, there\'re 3 kinds of resolution {i, c, f}\n        - for megadepth dataset, there\'re 4 kinds of resolution {i, i_resize, c, f}\n    '
    device = data['image0'].device
    (N, _, H0, W0) = data['image0'].shape
    (_, _, H1, W1) = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    (h0, w0, h1, w1) = (x // scale for x in [H0, W0, H1, W1])
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).expand(N, h0 * w0, 2)
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).expand(N, h1 * w1, 2)
    grid_pt1_i = scale1 * grid_pt1_c
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])
    (_, w_pt0_i) = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    (_, w_pt1_i) = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    def out_bound_mask(pt: Tensor, w: Tensor, h: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0
    loop_back = torch.stack([nearest_index0[_b][_i] for (_b, _i) in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0 * w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False
    conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)
    (b_ids, i_ids) = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})
    if len(b_ids) == 0:
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)
    data.update({'spv_b_ids': b_ids, 'spv_i_ids': i_ids, 'spv_j_ids': j_ids})
    data.update({'spv_w_pt0_i': w_pt0_i, 'spv_pt1_i': grid_pt1_i})

def compute_supervision_coarse(data: dict[str, Any], config: dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    if len(set(data['dataset_name'])) != 1:
        raise ValueError('Do not support mixed datasets training!')
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')

@torch.no_grad()
def spvs_fine(data: dict[str, Any], config: dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Update:\n        data (dict):{\n            "expec_f_gt": [M, 2]}\n    '
    (w_pt0_i, pt1_i) = (data['spv_w_pt0_i'], data['spv_pt1_i'])
    scale = config['LOFTR']['RESOLUTION'][1]
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2
    (b_ids, i_ids, j_ids) = (data['b_ids'], data['i_ids'], data['j_ids'])
    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius
    data.update({'expec_f_gt': expec_f_gt})

def compute_supervision_fine(data: dict[str, Any], config: dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError