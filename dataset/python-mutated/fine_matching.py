import math
import torch
import torch.nn as nn
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        if False:
            return 10
        super().__init__()

    def forward(self, feat_f0, feat_f1, data):
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            feat0 (torch.Tensor): [M, WW, C]\n            feat1 (torch.Tensor): [M, WW, C]\n            data (dict)\n        Update:\n            data (dict):{\n                'expec_f' (torch.Tensor): [M, 3],\n                'mkpts0_f' (torch.Tensor): [M, 2],\n                'mkpts1_f' (torch.Tensor): [M, 2]}\n        "
        (M, WW, C) = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        (self.M, self.W, self.WW, self.C, self.scale) = (M, W, WW, C, scale)
        if M == 0:
            assert self.training is False, 'M is always >0, when training, see coarse_matching.py'
            data.update({'expec_f': torch.empty(0, 3, device=feat_f0.device), 'mkpts0_f': data['mkpts0_c'], 'mkpts1_f': data['mkpts1_c']})
            return
        feat_f0_picked = feat_f0_picked = feat_f0[:, WW // 2, :]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1.0 / C ** 0.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)
        var = torch.sum(grid_normalized ** 2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized ** 2
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})
        self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        if False:
            while True:
                i = 10
        (W, scale) = (self.W, self.scale)
        mkpts0_f = data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]
        data.update({'mkpts0_f': mkpts0_f, 'mkpts1_f': mkpts1_f})