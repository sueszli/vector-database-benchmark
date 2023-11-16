import warnings
from abc import ABCMeta
from mmcv.runner import BaseModule

class BasePointNet(BaseModule, metaclass=ABCMeta):
    """Base class for PointNet."""

    def __init__(self, init_cfg=None, pretrained=None):
        if False:
            print('Hello World!')
        super(BasePointNet, self).__init__(init_cfg)
        self.fp16_enabled = False
        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @staticmethod
    def _split_point_feats(points):
        if False:
            while True:
                i = 10
        'Split coordinates and features of input points.\n\n        Args:\n            points (torch.Tensor): Point coordinates with features,\n                with shape (B, N, 3 + input_feature_dim).\n\n        Returns:\n            torch.Tensor: Coordinates of input points.\n            torch.Tensor: Features of input points.\n        '
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None
        return (xyz, features)