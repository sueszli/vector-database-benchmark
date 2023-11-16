import torch
import torch.nn as nn
from typing import Tuple
from ding.hpc_rl import hpc_wrapper

def shape_fn_scatter_connection(args, kwargs) -> list:
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Return shape of scatter_connection for hpc\n    Returns:\n        - shape (:obj:`list`): List like [B, M, N, H, W, scatter_type]\n    '
    if len(args) <= 1:
        tmp = list(kwargs['x'].shape)
    else:
        tmp = list(args[1].shape)
    if len(args) <= 2:
        tmp.extend(kwargs['spatial_size'])
    else:
        tmp.extend(args[2])
    tmp.append(args[0].scatter_type)
    return tmp

class ScatterConnection(nn.Module):
    """
    Overview:
        Scatter feature to its corresponding location
        In AlphaStar, each entity is embedded into a tensor,
        and these tensors are scattered into a feature map with map size.
    """

    def __init__(self, scatter_type: str) -> None:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Init class\n        Arguments:\n            - scatter_type (:obj:`str`): Supports ['add', 'cover']. If two entities have the same location, \\\n                scatter_type decides the first one should be covered or added to second one\n        "
        super(ScatterConnection, self).__init__()
        self.scatter_type = scatter_type
        assert self.scatter_type in ['cover', 'add']

    @hpc_wrapper(shape_fn=shape_fn_scatter_connection, namedtuple_data=False, include_args=[0, 2], include_kwargs=['x', 'location'], is_cls_method=True)
    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int], location: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            scatter x into a spatial feature map\n        Arguments:\n            - x (:obj:`tensor`): input tensor :math: `(B, M, N)` where `M` means the number of entity, `N` means                 the dimension of entity attributes\n            - spatial_size (:obj:`tuple`): Tuple[H, W], the size of spatial feature x will be scattered into\n            - location (:obj:`tensor`): :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)\n        Returns:\n            - output (:obj:`tensor`): :math: `(B, N, H, W)` where `H` and `W` are spatial_size, return the                scattered feature map\n        Shapes:\n            - Input: :math: `(B, M, N)` where `M` means the number of entity, `N` means                 the dimension of entity attributes\n            - Size: Tuple type :math: `[H, W]`\n            - Location: :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)\n            - Output: :math: `(B, N, H, W)` where `H` and `W` are spatial_size\n\n        .. note::\n\n            When there are some overlapping in locations, ``cover`` mode will result in the loss of information, we\n            use the addition as temporal substitute.\n        '
        device = x.device
        (B, M, N) = x.shape
        x = x.permute(0, 2, 1)
        (H, W) = spatial_size
        index = location[:, :, 1] + location[:, :, 0] * W
        index = index.unsqueeze(dim=1).repeat(1, N, 1)
        output = torch.zeros(size=(B, N, H, W), device=device).view(B, N, H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=index, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=index, src=x)
        output = output.view(B, N, H, W)
        return output

    def xy_forward(self, x: torch.Tensor, spatial_size: Tuple[int, int], coord_x: torch.Tensor, coord_y) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            scatter x into a spatial feature map\n        Arguments:\n            - x (:obj:`tensor`): input tensor :math: `(B, M, N)` where `M` means the number of entity, `N` means                the dimension of entity attributes\n            - spatial_size (:obj:`tuple`): Tuple[H, W], the size of spatial feature x will be scattered into\n            - coord_x (:obj:`tensor`): :math: `(B, M)` torch.LongTensor, each location should be x\n            - coord_y (:obj:`tensor`): :math: `(B, M)` torch.LongTensor, each location should be y\n        Returns:\n            - output (:obj:`tensor`): :math: `(B, N, H, W)` where `H` and `W` are spatial_size, return the                scattered feature map\n        Shapes:\n            - Input: :math: `(B, M, N)` where `M` means the number of entity, `N` means                the dimension of entity attributes\n            - Size: Tuple[H, W]\n            - Coord_x: :math: `(B, M)` torch.LongTensor, each location should be x\n            - Coord_y: :math: `(B, M)` torch.LongTensor, each location should be y\n            - Output: :math: `(B, N, H, W)` where `H` and `W` are spatial_size\n\n        note:\n            when there are some overlapping in locations, ``cover`` mode will result in the loss of information, we\n            use the addition as temporal substitute.\n        '
        device = x.device
        (B, M, N) = x.shape
        x = x.permute(0, 2, 1)
        (H, W) = spatial_size
        index = (coord_x * W + coord_y).long()
        index = index.unsqueeze(dim=1).repeat(1, N, 1)
        output = torch.zeros(size=(B, N, H, W), device=device).view(B, N, H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=index, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=index, src=x)
        output = output.view(B, N, H, W)
        return output