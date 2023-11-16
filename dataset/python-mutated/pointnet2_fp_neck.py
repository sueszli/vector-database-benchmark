from mmcv.runner import BaseModule
from torch import nn as nn
from mmdet3d.ops import PointFPModule
from ..builder import NECKS

@NECKS.register_module()
class PointNetFPNeck(BaseModule):
    """PointNet FP Module used in PointRCNN.

    Refer to the `official code <https://github.com/charlesq34/pointnet2>`_.

    .. code-block:: none

        sa_n ----------------------------------------
                                                     |
        ... ---------------------------------        |
                                             |       |
        sa_1 -------------                   |       |
                          |                  |       |
        sa_0 -> fp_0 -> fp_module ->fp_1 -> ... -> fp_module -> fp_n

    sa_n including sa_xyz (torch.Tensor) and sa_features (torch.Tensor)
    fp_n including fp_xyz (torch.Tensor) and fp_features (torch.Tensor)

    Args:
        fp_channels (tuple[tuple[int]]): Tuple of mlp channels in FP modules.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, fp_channels, init_cfg=None):
        if False:
            return 10
        super(PointNetFPNeck, self).__init__(init_cfg=init_cfg)
        self.num_fp = len(fp_channels)
        self.FP_modules = nn.ModuleList()
        for cur_fp_mlps in fp_channels:
            self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))

    def _extract_input(self, feat_dict):
        if False:
            while True:
                i = 10
        'Extract inputs from features dictionary.\n\n        Args:\n            feat_dict (dict): Feature dict from backbone, which may contain\n                the following keys and values:\n\n                - sa_xyz (list[torch.Tensor]): Points of each sa module\n                    in shape (N, 3).\n                - sa_features (list[torch.Tensor]): Output features of\n                    each sa module in shape (N, M).\n\n        Returns:\n            list[torch.Tensor]: Coordinates of multiple levels of points.\n            list[torch.Tensor]: Features of multiple levels of points.\n        '
        sa_xyz = feat_dict['sa_xyz']
        sa_features = feat_dict['sa_features']
        assert len(sa_xyz) == len(sa_features)
        return (sa_xyz, sa_features)

    def forward(self, feat_dict):
        if False:
            for i in range(10):
                print('nop')
        'Forward pass.\n\n        Args:\n            feat_dict (dict): Feature dict from backbone.\n\n        Returns:\n            dict[str, torch.Tensor]: Outputs of the Neck.\n\n                - fp_xyz (torch.Tensor): The coordinates of fp features.\n                - fp_features (torch.Tensor): The features from the last\n                    feature propagation layers.\n        '
        (sa_xyz, sa_features) = self._extract_input(feat_dict)
        fp_feature = sa_features[-1]
        fp_xyz = sa_xyz[-1]
        for i in range(self.num_fp):
            fp_feature = self.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)], sa_features[-(i + 2)], fp_feature)
            fp_xyz = sa_xyz[-(i + 2)]
        ret = dict(fp_xyz=fp_xyz, fp_features=fp_feature)
        return ret