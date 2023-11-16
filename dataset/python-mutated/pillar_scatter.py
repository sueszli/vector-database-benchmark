import torch
from mmcv.runner import auto_fp16
from torch import nn
from ..builder import MIDDLE_ENCODERS

@MIDDLE_ENCODERS.register_module()
class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, output_shape):
        if False:
            return 10
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    @auto_fp16(apply_to=('voxel_features',))
    def forward(self, voxel_features, coors, batch_size=None):
        if False:
            return 10
        'Forward function to scatter features.'
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)

    def forward_single(self, voxel_features, coors):
        if False:
            return 10
        'Scatter features of single sample.\n\n        Args:\n            voxel_features (torch.Tensor): Voxel features in shape (N, C).\n            coors (torch.Tensor): Coordinates of each voxel.\n                The first column indicates the sample ID.\n        '
        canvas = torch.zeros(self.in_channels, self.nx * self.ny, dtype=voxel_features.dtype, device=voxel_features.device)
        indices = coors[:, 2] * self.nx + coors[:, 3]
        indices = indices.long()
        voxels = voxel_features.t()
        canvas[:, indices] = voxels
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return canvas

    def forward_batch(self, voxel_features, coors, batch_size):
        if False:
            for i in range(10):
                print('nop')
        'Scatter features of single sample.\n\n        Args:\n            voxel_features (torch.Tensor): Voxel features in shape (N, C).\n            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).\n                The first column indicates the sample ID.\n            batch_size (int): Number of samples in the current batch.\n        '
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = torch.zeros(self.in_channels, self.nx * self.ny, dtype=voxel_features.dtype, device=voxel_features.device)
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny, self.nx)
        return batch_canvas